from pyjiting.types import double64_t, int64_t, make_array_type, str_t
from conftest import verified_module


def test_generated_modules_verify_for_control_flow_and_runtime_guards():
    module = verified_module('''
        def kernel(start, step):
            total = 0
            for value in range(start, 0, step):
                if value % 2:
                    total += value
            return total
    ''', [int64_t, int64_t])

    ir_text = str(module)
    assert 'runtime_error' in ir_text
    assert 'for_latch' in ir_text


def test_generated_array_module_verifies_for_strided_multidimensional_access():
    module = verified_module('''
        def update(values, row, col):
            values[row, col] += 1.5
            return values[row, col] / 2
    ''', [make_array_type(double64_t), int64_t, int64_t])

    ir_text = str(module)
    assert 'pyjiting.ndarray.double' in ir_text
    assert 'getelementptr i8, i8*' in ir_text
    assert 'align 1' in ir_text


def test_string_index_and_iteration_load_utf32_codepoints_in_native_ir():
    module = verified_module('''
        def first(value):
            for character in value:
                return character
            return ''
    ''', [str_t])

    ir_text = str(module)
    assert 'getelementptr i32, i32*' in ir_text
    assert 'foreach_body' in ir_text
