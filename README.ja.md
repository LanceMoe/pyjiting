**Languages:** [English](README.md) · [简体中文](README.zh.md) · [日本語](README.ja.md)

# Pyjiting

Pyjiting は、卒業論文（学士学位論文）の成果物として開発された実験的な Python 用 JIT コンパイラです。LLVM（[llvmlite](https://llvmlite.readthedocs.io/) 経由）をバックエンドとして採用し、軽量でコンパクトな汎用 Python JIT コンパイラの実装を目指しています。

```python
from pyjiting import jit

@jit
def fib(x):
    if x < 3:
        return 1
    return fib(x-1) + fib(x-2)

print(fib(40))  # CPython より約 40 倍高速

```

## 仕組み

Pyjiting は、関数が最初に呼び出されたタイミングでネイティブコードへ JIT コンパイルします。`@jit` デコレータが付与された関数は、次のようなパイプラインを経て実行されます。

```
 Python ソース
      │  ast.parse + ASTVisitor (pyjiting/parser.py)
      ▼
 Core AST (pyjiting/ast.py)
      │  Hindley-Milner 風の型推論 (pyjiting/infer.py)
      ▼
 型制約 (pyjiting/types.py, pyjiting/utils.py)
      │  呼び出しごとに実引数の型とユニファイ
      ▼
 特化された AST ──► LLVM IR コード生成 (pyjiting/codegen.py)
                          │
                          ▼
              LLVM 最適化（O3 + ループベクトル化、New Pass Manager）
                          │
                          ▼
              MCJIT コンパイル ──► ctypes コールバック (pyjiting/ll_types.py)

```

1. **パース** —— 関数ソースを解析し、独自のコンパクトな Core AST へ変換します（`pyjiting/parser.py`）。
2. **型推論** —— Hindley-Milner 風の推論器が型制約を収集・解決します（`pyjiting/infer.py`）。
3. **特化（Specialization）** —— 関数呼び出し時に実引数の型（`int`、`float`、`numpy.ndarray` など）と推論結果をユニファイし、具体的なモノモーフィック（単一型）シグネチャを導出します（`pyjiting/main.py`）。
4. **コード生成** —— そのシグネチャに対応する LLVM IR を生成し、LLVM 最適化パスおよび MCJIT エンジンを通じてネイティブコードへコンパイルします。
5. **キャッシュ** —— 特化された各シグネチャはマングル名で識別され、初回のみコンパイルされてキャッシュされます。以降、同一型での呼び出しはネイティブコードへ直接ディスパッチされます。

## サポート対象サブセット

| 領域 | サポートされる挙動 | 回帰テスト |
| --- | --- | --- |
| 特化 | スカラ／配列／文字列シグネチャごとの特化、決定論的な命名、並行コンパイルの分離、ネイティブ JIT 関数間呼び出し | `tests/test_specialization.py`、`tests/test_extensions.py` |
| スカラ | `bool`、`int32`、`int64`、`float32`、`float64`。決定論的な型昇格および固定幅整数のラップアラウンド | `tests/test_infer.py`、`tests/test_arith.py` |
| 演算 | `+ - * / // % **`、単項 `-`、Python 互換の floor/mod 符号規則、定数整数べき乗、NaN を考慮した真理値判定 | `tests/test_arith.py` |
| 制御フロー | `if`、`while`、`range`、一次元配列／文字列の反復、`break`、`continue`、ループ `else` | `tests/test_control_flow.py`、`tests/test_extensions.py` |
| 文字列 | Unicode 値、比較／包含判定、完全なスライス、連結／反復、変換、文字種判定、検索、`ord`／`chr` | `tests/test_string.py`、`tests/test_string_phase2.py` |
| タプル | 固定長と可変長、動的添字、スライス、連結、反復演算、反復、比較、アンパック | `tests/test_tuple_phase4.py`、`tests/test_builtin_types.py` |
| その他の組み込み型 | `complex`、`bytes`、`bytearray`、`memoryview`、`range`、`slice`、`None`、オプショナル型注釈 | `tests/test_builtin_types.py` |
| リスト | 異種および入れ子の要素、リテラル、内包表記、`list()`、添字とスライスによる読み書き・削除、反復、標準メソッド、引数と戻り値の同一性 | `tests/test_list.py` |
| 集合 | `set` と `frozenset`、リテラルと内包表記、反復、集合演算、比較、標準メソッド | `tests/test_set.py` |
| 辞書 | `**` 展開を含むリテラル、内包表記、`dict()`、キーによる読み書き、動的ビュー、マージ演算、標準メソッド | `tests/test_dict.py` |
| 配列 | 4 種の数値 dtype、安全な添字、多次元 Strided アクセス、一次元反復、多次元全体の `sum`／`any`／`all` スカラーリダクション | `tests/test_array.py`、`tests/test_extensions.py`、`tests/test_numeric_phase3.py` |
| Python 構文 | f 文字列、キーワード／可変長引数、CPython で実行するジェネレータ、例外処理、コンテキスト管理、パターンマッチ、非同期関数とクロージャ | `tests/test_python_syntax.py` |
| 組み込み関数 | Python ランタイムによる `round`、`pow`、`min`／`max`、`sorted`、`map`、`filter`、`zip` などと、ネイティブの数値・文字列演算 | `tests/test_python_builtins.py`、`tests/test_numeric_phase3.py` |
| 定数 | `@jit` 適用時に不変スカラ／文字列のグローバルおよびクロージャ値をキャプチャ | `tests/test_numeric_phase3.py` |
| アノテーションとコールバック | スカラ／文字列アノテーション、`np.ndarray` dtype の遅延特化、型注釈付き `@reg` | `tests/test_parser.py`、`tests/test_reg_callback.py`、`tests/test_extensions.py` |
| 検証 | パーサのソース位置保持、推論ルール検証、生成された LLVM モジュールのバリデーション | `tests/test_parser.py`、`tests/test_codegen.py` |

### Python の型とコンテナ

リストは異なる型の要素、入れ子のリスト、`None` を保持できます。負の添字、スライス、要素やスライスへの代入および削除、starred および入れ子のアンパック、反復、包含判定、比較、連結、反復演算、`sort(key=..., reverse=...)` を含むすべての標準リストメソッドをサポートします。引数や戻り値には Python のリストそのものを使います。同じリストを参照する変数には変更が反映され、スライスと `copy()` は浅いコピーを作ります。`list[T]` はリストを表し、実行時に各要素の型は検査しません。関数の制御フローは LLVM がコンパイルし、リスト操作は Python ランタイムが実行します。

集合と辞書はコンストラクタ、内包表記、反復、演算、標準メソッドに対応します。辞書の挿入順と動的ビューは維持され、変更は呼び出し側から参照できます。

JIT は `complex`、`bytes`、`bytearray`、`memoryview`、`range`、`slice`、`None` を受け取り、返せます。複素数演算、バイナリシーケンスのメソッド、バッファビュー、範囲とスライスの操作は Python に従います。バッファとビューの同一性を保ち、書き込み可能なビューの変更は元のバッファに反映されます。`split()`、`encode()`、`format()` などの文字列メソッドも Python で実行できます。

`int | None`、`Optional[int]`、`object` などの注釈は、実行時に Python オブジェクトを使用します。共用体の各メンバー型は検査しません。固定タプルの要素を含め、真偽値は `bool` 型で返します。ネイティブの `int` は符号付き 64 ビットで、演算がオーバーフローするとラップします。`object` と注釈した引数は Python の算術を使います。ネイティブの NumPy 配列は数値型 4 種類に対応し、直接返すことはできません。

### 数値と配列のセマンティクス

Pyjiting は、任意精度の Python 整数ではなく、固定幅の Int32/Int64 を使用します。スカラの混在演算は決定論的な昇格規則に従います。int32 は必要に応じて int64 へ昇格し、float32 は int64 または float64 と演算された場合に float64 へ昇格します。`/` 演算は常に浮動小数点数を返します。代入処理では安全な型昇格のみが許可されるため、通常の Python `int` を int32 配列に代入する場合は、あらかじめ `np.int32` へ明示的にキャストする必要があります。

整数演算は 2 の補数表現による固定幅の挙動に準拠します。特に、符号付き最小整数を -1 で割った場合でも任意精度への昇格は行われず、オーバーフローして符号付き最小整数のままとなります。

配列ディスクリプタは `data`、`ndim`、`shape`、バイト単位の `strides`、`itemsize`、NumPy flags、元の配列への参照を保持します。`list(array)` では NumPy の反復の挙動が維持され、`[array]` では配列の同一性が維持されます。要素と shape の負添字および境界検査に対応し、転置、slice、負 stride、バイト stride、未アライン view も安全に処理します。実際に読み取り専用配列へ書き込むと `ValueError("assignment destination is read-only")` になります。範囲外は `IndexError`、添字数の不一致は `ValueError` です。配列生成、ブロードキャスト、配列の戻り値、配列全体の ufunc は未対応です。
任意ランクの Strided 配列ではネイティブな全体 `sum`、`any`、`all` を利用でき、int32/float32 の和は int64/float64 へ拡張され、空配列の単位元は Python と一致します。axis リダクションは未対応です。

不変の数値／文字列グローバルおよびクロージャ値は `@jit` 適用時に Core リテラルとして固定されます。ネイティブ `math` は `sin`、`cos`、`sqrt`、`exp`、`log`、`log2`、`log10`、浮動小数点分類、標準定数に対応し、domain/range エラーは JIT エラー ABI で伝播します。

文字列は長さ付き UTF-32 ABI を使用し、Unicode コードポイントと埋め込み NUL を保持します。一時値と戻り値は呼び出し単位の arena で管理されます。スライスは省略、正、負、動的な非ゼロ step に対応し、包含判定、Unicode 大小文字変換、空白除去、置換、文字種判定、`ord`／`chr` は静的サブセット内で Python の意味論に従います。

固定タプルは要素型に応じた構造体で特化し、元の Python タプルがある場合はその参照を保持します。`tuple` と `tuple[T, ...]` の注釈は Python タプルを使います。動的添字、スライス、反復、スター付きアンパック、比較、シーケンス演算に対応します。

各特化コードは、デコレートされた Python 関数の識別情報（identity）と型シグネチャから生成される一意の LLVM シンボルを使用します。そのため、同じ関数名を持つ別々の関数が誤ってキャッシュを共有してしまうことはありません。

`@jit` と `jit_from_source` は位置専用・キーワード専用引数、対応する型のデフォルト値、`*args` と `**kwargs` を扱えます。変更可能なデフォルト値は呼び出し間で同じオブジェクトです。コンパイル済みの関数からキーワードや展開した引数を渡せます。ソースから作る関数のデフォルト値は一度だけ評価され、関数作成後に追加された名前も含めて、渡された `namespace` 辞書からグローバル名を解決します。

LLVM 関数内で f 文字列を利用でき、変換とネストした書式指定は Python の書式化プロトコルを使います。`round`、`pow`、`bin`、`oct`、`hex`、`ascii`、`callable`、`getattr`、`hasattr`、`map`、`filter`、`print` などは Python ランタイムを使います。`min` と `max` は `key` と `default`、`sorted` は `key` と `reverse`、`zip` は `strict` に対応します。

ジェネレータ式、`yield`、`yield from`、`try`／`except`／`finally`、`with`、`match`、非同期構文、ネストした定義、ラムダ、または `locals()` など呼び出しフレームに依存する組み込み関数を直接使う場合は、関数全体を CPython で実行します。通常の JIT 入口から使え、LLVM 関数からも呼び出せます。`runtime_stats(compiled)['backend']` は `'llvm'` または `'cpython'` を返し、`python_features` で CPython の選択理由を確認できます。

LLVM 関数では `compiled.specialize(*args, **kwargs)` で実行せずにコンパイルでき、`runtime_stats(compiled)` と `inspect_specializations(compiled)` で統計を、`get_llvm_ir(compiled, *args, **kwargs)` で IR を確認できます。統計にはコンパイル待機、シグネチャごとのコンパイル回数、失敗情報、文字列ランタイムと登録コールバックの呼び出しが含まれます。`JITContext` は独立したエンジン、モジュールと特化の上限、明示的なクローズ、および登録コールバックの解放を提供します。

`fallback=True` では frontend、型推論、codegen が未対応の経路だけ元の Python 関数を実行します。`FallbackWarning` は `function`、`reason`、`error_type` を公開し、`fallback_warning` は既定の `"once"`、`"always"`、`"ignore"` から選べます。関数の特化上限、context のリソース上限、close 済み runtime、LLVM 内部エラーは fallback しません。文字列の index と比較はネイティブ UTF-32 経路です。`upper()`、`lower()` など複雑な Unicode 変換は引き続き Python runtime を呼び、その回数は `runtime_stats()['string_callbacks']` で確認できます。

配列結果を生成する kernel では内部 descriptor を返さず、呼び出し側が所有する output 配列を渡してください。`examples/example_array_output.py` を参照してください。shape/dtype の互換性と書き込み可能な output が必要です。独立 output、完全な in-place、同じ base を共有する非重複 view をサポートします。部分的に重なる input/output view は、順次書き込みが後続読み取りを変えるため契約外です。

## 動作要件

* Python >= 3.12
* [uv](https://docs.astral.sh/uv/)
* llvmlite >= 0.49.0
* numpy >= 2.5.2

依存関係は `pyproject.toml` に定義され、`uv.lock` で固定されています。開発環境の構築やコマンドの実行には `uv` を使用してください（`requirements.txt` は開発ワークフローで使用しません）。

## uv による環境構築と開発

```bash
uv sync --extra dev
uv run coverage run -m pytest -q
uv run coverage report
uv run pyright

```

`uv sync` を実行すると、ロックファイルに基づいてローカルの `.venv` が作成・更新されます。プロジェクトが編集可能モードでインストールされるため、`uv run` 経由で直接サンプルスクリプトを実行できます。

パッケージのビルド手順：

```bash
uv build

```

ソース配布物（sdist）および wheel ファイルが `dist/` ディレクトリに生成されます。

## 使い方

### 基本的な JIT コンパイル

```python
from pyjiting import jit

@jit
def is_prime(x):
    if x < 2:
        return 0
    i = 2
    while i * i <= x:
        if x % i == 0:
            return 0
        i = i + 1
    return 1

is_prime(169941229)   # 1 を返す（int 用に特化コンパイル）

```

### 動的型付けへの対応（多相化）

同じ関数であっても、呼び出し時の引数の型に応じてそれぞれ特化されます。新しい型の組み合わせが渡された初回のみ追加コンパイルが走り、以降はキャッシュから即座に実行されます。

```python
@jit
def test(a, b):
    return a + b

print(test(114, 514))    # 628   (int64 特化)
print(test(11.4, 51.4))  # 62.8  (double/float64 特化)

```

### `@jit` と `@reg`

* **`@jit`**: 関数を JIT コンパイル対象として指定します。関数定義時に AST がパースされ、初回呼び出し時の引数型シグネチャに応じてネイティブコードへコンパイルされます。数値計算やループ、再帰処理などの重い処理に最適です。
* **`@reg`**: 通常の Python 関数を、JIT コード内から呼び出し可能なコールバック関数として登録します。JIT 側ではコンパイルされず、ネイティブ実行時に ctypes 境界を跨いで元の Python 関数を呼び出します。ログ出力、進捗表示、既存の Python ライブラリとの連携に適しています。登録対象にはサポートされたスカラまたは文字列型の型アノテーション（`int`、`float`、`bool`、`str`、`np.int32`、`np.int64`、`np.float32`、`np.float64`）が必要です。同名のコールバックは callable の同一性で分離されます。コールバック例外は dispatch frame が保持し、最外層の Python 呼び出し地点で再送出されるため、ネイティブのユーザーロジックがダミーの戻り値で実行を続けることはありません。

| デコレータ | 役割 | 実行形態 | 主な用途 |
| --- | --- | --- | --- |
| `@jit` | 関数をコンパイルする | LLVM が生成したネイティブコード | 数値計算カーネル、ループ、再帰 |
| `@reg` | JIT から呼べる callable を登録 | ctypes 経由で元の Python 実装を実行 | ログ出力、画面表示、Python 依存処理の統合 |

### JIT 関数から Python 関数をコールバックする

`@reg` で登録し、スカラ型の型アノテーションを完全につけた Python 関数は、JIT ネイティブコードからシームレスに呼び出すことができます。

```python
from pyjiting import jit, reg

@reg
def report(x: int) -> int:
    print(f'{x} is prime!')
    return 0

@jit
def find_primes(n):
    for i in range(2, n):
        is_prime = True
        for j in range(2, i):
            if i % j == 0:
                is_prime = False
                break
        if is_prime == True:
            report(i)
    return 0

```

### サンプルコードの実行

```bash
uv run examples/example_fib.py
uv run examples/example_find_primes.py
uv run examples/example_is_prime.py
uv run examples/example_loop.py
uv run examples/example_while.py
uv run examples/example_generic.py
uv run examples/example_bool_ops.py
uv run examples/example_float_math.py
uv run examples/example_array_2d.py
uv run examples/example_annotations.py
uv run examples/example_mixed_types.py
uv run examples/example_lists.py
uv run examples/example_sets.py
uv run examples/example_dicts.py
uv run examples/example_builtin_types.py
uv run examples/example_python_syntax.py

```

## パフォーマンス

`uv run benchmarks/benchmark.py --repeat 20` で cold、warm、CPython、NumPy の比較を複数回実行できます。`--json result.json` を追加すると、生サンプルと環境メタデータを保存します。各結果を検証し、median、minimum、standard deviation を報告します。cold 呼び出しは特化と LLVM コンパイルを含み、warm 呼び出しは既存のネイティブ特化を使用します。

サンプルコードは `examples/` ディレクトリに収録されています。

```
テスト環境:
CPU: 13th Gen Intel(R) Core(TM) i5-13600K @ 5.4GHz
メモリ: 64GB
OS: Windows 11 64bit (10.0.26200)
Python 3.12.11 64bit
LLVMLite 0.49.0
NumPy 2.5.2

```

| ベンチマーク | JIT | CPython | 高速化倍率 |
| --- | --- | --- | --- |
| `fib(40)` | 126.2 ms | 5056.4 ms | ~40.1x |
| `find_primes(100000)` | 909.4 ms | 11984.9 ms | ~13.2x |
| `is_prime(169941229)` | 340.5 ms | 4462.4 ms | ~13.1x |
| `loop(100000000)` | ~0 ms | 1871.1 ms | ∞（※下記注釈参照） |
| `test_while(100000000)` | ~0 ms | 1669.1 ms | ∞（※下記注釈参照） |

実行ログ出力:

```
fib_jit(40) = 102334155 (cost time: 126.23047828674316 ms)
fib_nojit(40) = 102334155 (cost time: 5056.420564651489 ms)
rate: 40.057049876380916

find_primes_jit(100000) = 0 (cost time: 909.3630313873291 ms)
find_primes_nojit(100000) = 0 (cost time: 11984.926223754883 ms)
rate: 13.179473774594307

is_prime_jit(169941229) = True (cost time: 340.4872417449951 ms)
is_prime_nojit(169941229) = True (cost time: 4462.382793426514 ms)
rate: 13.105873719546224

loop_jit(100000000) = 200000000 (cost time: 0.0 ms)
loop_nojit(100000000) = 200000000 (cost time: 1871.0994720458984 ms)
rate: Infinite

test_while_jit(100000000) = 100000000 (cost time: 0.0 ms)
test_while_nojit(100000000) = 100000000 (cost time: 1669.1172122955322 ms)
rate: Infinite

```

> **※注記**: 単純な `loop` / `while` ベンチマークはカウンタを加算するだけの処理であるため、LLVM の `-O3` 最適化によってループ自体が定数畳み込み等で完全に除去されます（そのため見かけ上の高速化倍率が "Infinite" となります）。その他のベンチマークは実際に有意なループ演算を行っています。

## プロジェクト構成

```
pyjiting/
├── __init__.py   # jit, reg のエクスポート
├── main.py       # @jit / @reg デコレータ、特化管理およびキャッシュ機構
├── parser.py     # Python AST -> Core AST への変換
├── ast.py        # Core AST ノード定義
├── infer.py      # Hindley-Milner 風の型推論器
├── codegen.py    # Core AST -> LLVM IR コード生成
├── ll_types.py   # ctypes ラッパー、名前マングリング、引数マーシャリング
├── types.py      # 型システム定義 (BaseType, VarType, FuncType, ArrayType など)
└── utils.py      # 単一化（Unification）および型代入ユーティリティ

```

## 制限事項

本プロジェクトは研究・教育を目的としたプロトタイプ実装であり、プロダクション環境での利用を想定したものではありません。主な制限事項は以下の通りです。

* LLVM コンパイルは静的に型付けできる Python の一部を対象とします。CPython で実行する関数には LLVM の特化や IR はありません。
* `for` は `range`、リスト、集合、辞書、タプル、文字列、バイナリシーケンス、一次元配列に対応します。コンパイル時に range の step が 0 と分かる場合はエラーになります。
* 実行時のゼロ除算は `ZeroDivisionError` を送出します。動的にステップ数が 0 となった range は、JIT のエラーハンドリング ABI 経由で `ValueError` を送出します。
* 具体的な戻り型の関数は、すべての制御フローパスで値を返します。オプショナル型、共用体、オブジェクト型の場合は暗黙に `None` を返せます。
* 整数のべき乗（`**`）は、指数部がコンパイル時定数である必要があります（動的な負の指数は静的に戻り型を一意に決定できないため）。
* `@reg` 関数はサポート対象のスカラ型または文字列型アノテーションを完全に備える必要があります。コールバック例外は最外層の Python 呼び出し地点で再送出され、ctypes ABI を跨ぎません。
* LLVM 関数にはグローバルオブジェクト、型注釈、多次元配列の反復、axis リダクション、NumPy 配列全体の演算に制約があります。

# 謝辞

本プロジェクトは、[numpile](https://dev.stephendiehl.com/numpile/) のチュートリアルに着想を得て、その成果を基盤として発展させたものである。

また、多大なるご示唆と的確なご指導を賜りました小笠原武史教授（Prof. Takeshi Ogasawara）に、心より深く感謝申し上げます。
