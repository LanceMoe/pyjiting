import llvmlite.binding as llvm
from llvmlite import ir

from pyjiting.codegen import LLVMCodeGen
from pyjiting.infer import TypeInferencer
from pyjiting.parser import ASTVisitor
from pyjiting.registry import signatures


def parse_function(source):
    return ASTVisitor()(source)


def infer_function(source, arg_types):
    tree = parse_function(source)
    signature = TypeInferencer(arg_types, signatures()).visit(tree)
    return tree, signature


def verified_module(source, arg_types):
    tree, signature = infer_function(source, arg_types)
    module = ir.Module(name='pyjiting.test')
    LLVMCodeGen(module, signature.return_type, arg_types).visit(tree)
    binding_module = llvm.parse_assembly(str(module))
    binding_module.verify()
    return module
