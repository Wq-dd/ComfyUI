import inspect
import pickle
import json
import functools

import torch
import numpy as np
import onnx

import tvm
from tvm import relay
from tvm.contrib.hexaflake.aigpu_utils import hexaflake_aigpu_bfloat16_func_register
from tvm.contrib.hexaflake.runtime.tc_graph_executor import TcGraphModule


device = tvm.device("aigpu", 0)
torch_cpu_device = torch.device("cpu")



def export_models(model, func, kwargs):

    tl_kwargs, all_kwargs, keyword_lst_prms = kwargs
    stat_kwargs = align_bs_stat(model, tl_kwargs)

    def dec_func_sig(func):
        oldsig = inspect.signature(func)
        # search if a VAR_POSITIONAL or VAR_KEYWORD is present
        # if yes insert step parameter before it, else insert it in last position
        params = list(oldsig.parameters.values())
        names = stat_kwargs.keys()
        for name in names:
            newparam = inspect.Parameter(name,
                                        inspect.Parameter.POSITIONAL_OR_KEYWORD)
            params.append(newparam)
            # we can now build the signature for the wrapper function
        sig = oldsig.replace(parameters = params)

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            bound = sig.bind(self, *args, **kwargs) # compute the bound parameter list
            bound.apply_defaults()
            for name in names:
                v = bound.arguments[name]
                del bound.arguments[name]
                setattr(self, name, v)
            cr = func(*bound.args, **bound.kwargs)
            return cr
        wrapper.__signature__ = sig
        return wrapper

    @dec_func_sig
    def trace_func(self):
        def legalize_args_kind(t_kwargs):
            # merge list
            if len(keyword_lst_prms) > 0:
                for k, v in keyword_lst_prms.items():
                    t_kwargs[k] = []
                    for name in v:
                        elem = t_kwargs.pop(name)
                        t_kwargs[k].append(elem)

        def legalize_output(model, rt):
            if torch.is_tensor(rt):
                return rt
            tup = rtn_obj2tup(rt)
            with open(model.tvm_comp['rtn_obj'], 'wb') as fp:
                pickle.dump(rt, fp)
            return tup

        _kwargs = {}
        for name in stat_kwargs.keys():
            _kwargs.update({name : getattr(self, name)})
        legalize_args_kind(_kwargs)
        # split args to positional and key-word prms
        _kwargs = compl_args(_kwargs, all_kwargs)
        rt = func(**_kwargs)
        rt = legalize_output(self.dummy_model, rt)
        return rt

    # torch module's trace relay ir include module prms var, func's not include.
    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.dummy_model = model
    Wrapper.forward = trace_func

    with torch.no_grad():
        name = model.tvm_conf['rt_name']
        print(f"tvm tracing {name} ...")
        names, dummys = list(stat_kwargs.keys()), list(stat_kwargs.values())
        shapes = [a.shape for a in dummys]
        _dtypes = [a.dtype for a in dummys]
        randfunc = lambda dt: torch.randn if dt == torch.float32 else torch.ones
        dummys = [randfunc(_dtypes[i])(sp, dtype=_dtypes[i], device=devices.device) for i, sp in enumerate(shapes)]

        infos = [(a, b) for a, b in zip(names, shapes)]
        info_dct = dict(zip(names, zip(shapes, [str(d) for d in _dtypes])))
        info_json = json.dumps(info_dct, sort_keys=False, indent=4, separators=(',', ':'))
        prefix = model.tvm_comp['pref']
        with open(f"{prefix}_info.json", "w") as fp:
            fp.write(info_json)
        if is_debug(model):
            print('kwarg_inputs', tuple(zip(infos, _dtypes)))
        kwarg_inps = dict(zip(names, dummys))
        model.eval()
        traced = torch.jit.trace(
            Wrapper(model),
            check_trace=False,
            example_kwarg_inputs=kwarg_inps
        )
        if is_debug(model):
            debug_trace(traced, prefix)
        mod, params = relay.frontend.from_pytorch(
            traced,
            infos
        )
        return mod, params


def compile_text_encoder():
    # text_encoder = torch.jit.load("output/pt/text_encoder.onnx")
    # inp_dict = [("input_tokens", [1, 77, 768]), ("attention_mask", [77, 77])]
    # mod, params = relay.frontend.from_pytorch(text_encoder, inp_dict)
    text_encoder = onnx.load("output/pt/text_encoder.onnx")
    inp_dict = {"x": [1,77,768], "mask.1": [77,77]}
    mod, params = relay.frontend.from_onnx(text_encoder, inp_dict)
    mod = tvm.IRModule.from_expr(mod["main"].body)
    mod = relay.transform.FoldConstant()(mod)
    target = {"aigpu": "aigpu", "cpu": "llvm"}
    configs = {
                "aigpu.backend.config": {},
                "relay.fallback_device_type": device.device_type,
                "relay.ToMixedPrecision.mixed_precision_type": "float32",
                "relay.SimplifyInference.ignored_ops": "nn.layer_norm;nn.gelu;nn.instance_norm"
            }
    hexaflake_aigpu_bfloat16_func_register()
    ctxt = tvm.transform.PassContext(opt_level=3, config=configs)  # 3 for aigpu, 2 for cpu
    with ctxt:
        print("=========start to build=========")
        lib = relay.build(mod, target=target, target_host="llvm", params=params)
        print("done!")
    return


def np_bf162np_float(arr):
    """Convert a numpy array of bf16 (uint16) to a numpy array
    of float"""
    u32 = np.left_shift(arr.astype("uint32"), 16)
    return u32.view("<f4")


def torch_from_numpy_lst(np_lst):
    arr = []
    for out in np_lst:
        if out.dtype == 'uint16':
            out = np_bf162np_float(out)
        out = torch.from_numpy(out).to(torch_cpu_device)
        arr.append(out)
    return arr


def compile_text_emb():
    emb = torch.jit.load("output/pt/emb.pt")
    inp_dict = [("input_tokens", ([1, 77], "int"))]
    mod, params = relay.frontend.from_pytorch(emb, inp_dict)
    mod = tvm.IRModule.from_expr(mod["main"].body)
    mod = relay.transform.FoldConstant()(mod)
    target = {"aigpu": "aigpu", "cpu": "llvm"}
    configs = {
                "aigpu.backend.config": {},
                "relay.fallback_device_type": device.device_type,
                "relay.ToMixedPrecision.mixed_precision_type": "bfloat16",
                "relay.SimplifyInference.ignored_ops": "nn.layer_norm;nn.gelu;nn.instance_norm"
            }
    hexaflake_aigpu_bfloat16_func_register()
    ctxt = tvm.transform.PassContext(opt_level=3, config=configs)  # 3 for aigpu, 2 for cpu
    with ctxt:
        lib = relay.build(mod, target=target, target_host="llvm", params=params)
        print("done!")
        rt_mod = TcGraphModule(lib, device) # infer instance
        dev_host = tvm.aigpu_host(0)
        data = np.int32(np.load("output/data.npy"))
        data = tvm.nd.array(data, dev_host)
        rt_mod.set_input("input_tokens", data)
        rt_mod.run()
        outs = rt_mod.get_output_numpy()
        if isinstance(outs, list):
            outs = torch_from_numpy_lst(outs)
        elif outs.dtype == 'uint16':
            outs = torch.from_numpy(np_bf162np_float(outs)).to(torch_cpu_device)
        else:
            outs = torch.from_numpy(outs).to(torch_cpu_device)   
        
        # num_outs = rt_mod.get_num_outputs()
        # outs = []
        # for i in range(num_outs):
            # t_out = rt_mod.get_output(i).numpy()
            # if t_out.dtype.name == "uint16":
                # t_out = np_bf162np_float(t_out)
            # outs.append(t_out)
        return outs


def compile_cpu():
    emb = torch.jit.load("output/pt/emb.pt")
    shape_list = [("input_tokens", ([1, 77], "int"))]
    mod, params = relay.frontend.from_pytorch(emb, shape_list)    
    target = tvm.target.Target("llvm", host="llvm")
    dev = tvm.cpu(0)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)    

    from tvm.contrib import graph_executor

    dtype = "float32"
    m = graph_executor.GraphModule(lib["default"](dev))
    # 设置输入
    m.set_input("input_tokens", tvm.nd.array(np.int32(np.load("output/data.npy")), dev))
    # 执行
    m.run()
    # 得到输出
    tvm_output = m.get_output(0).numpy()
    temp = np.load("output/x.npy")
    return


def compile_vae_decoder():
    text_encoder = torch.jit.load("output/pt/vae_decoder.pt")
    inp_dict = [("dec", ((1, 4, 64, 64), "float"))]
    mod, params = relay.frontend.from_pytorch(text_encoder, inp_dict)
    
    # text_encoder = onnx.load("output/pt/vae_decoder.onnx")
    # inp_dict = {"dec": [1,4,64,64]}
    # mod, params = relay.frontend.from_onnx(text_encoder, inp_dict)

    mod = tvm.IRModule.from_expr(mod["main"].body)
    mod = relay.transform.FoldConstant()(mod)
    target = {"aigpu": "aigpu", "cpu": "llvm"}
    target = {"cpu": "llvm"}
    configs = {
                "aigpu.backend.config": {},
                "relay.fallback_device_type": device.device_type,
                "relay.ToMixedPrecision.mixed_precision_type": "float32",
                "relay.SimplifyInference.ignored_ops": "nn.layer_norm;nn.gelu;nn.instance_norm"
            }
    hexaflake_aigpu_bfloat16_func_register()
    # ctxt = tvm.transform.PassContext(opt_level=3, config=configs)  # 3 for aigpu, 2 for cpu
    ctxt = tvm.transform.PassContext(opt_level=3)  # 3 for aigpu, 2 for cpu
    with ctxt:
        print("=========start to build=========")
        lib = relay.build(mod, target=target, target_host="llvm", params=params)
        print("done!")
    return



if __name__ == "__main__":
    # compile_cpu()
    compile_vae_decoder()
    # compile_text_encoder()
    # compile_text_emb()    