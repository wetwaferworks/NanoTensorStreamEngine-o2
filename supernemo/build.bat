@echo off
rem SuperNemo pieces (pure C99, no locks). Run from this folder.
setlocal
set F=-O2 -std=c99 -Wall -Wextra -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L -ffp-contract=off
set L=..\littlenemo\src
gcc %F% -o test_sn_moe.exe tests\test_sn_moe.c src\sn_moe.c -lm || exit /b 1
gcc %F% -o sn_packcheck.exe tests\sn_packcheck.c src\sn_pack.c -lm || exit /b 1
gcc %F% -o sn_preflight.exe tests\sn_preflight.c src\sn_pack.c -lm || exit /b 1
gcc %F% -o sn_mktiny.exe tests\sn_mktiny.c -lm || exit /b 1
gcc %F% -o test_sn_model.exe tests\test_sn_model.c src\sn_model.c src\sn_pack.c src\sn_pqk.c src\sn_moe.c src\sn_gemm.c src\sn_cuda.c %L%\mamba.c %L%\attn.c %L%\pool.c -lm || exit /b 1
gcc %F% -o sn_mkvis.exe tests\sn_mkvis.c -lm || exit /b 1
gcc %F% -o test_sn_vision.exe tests\test_sn_vision.c src\sn_vision.c src\sn_gemm.c src\sn_pack.c %L%\pool.c -lm || exit /b 1
gcc %F% -o test_sn_image.exe tests\test_sn_image.c src\sn_image.c -lm || exit /b 1
gcc %F% -o test_sn_cuda.exe tests\test_sn_cuda.c src\sn_cuda.c src\sn_pack.c src\sn_pqk.c -lm || exit /b 1
gcc %F% -o sn-engine.exe src\sn_engine.c src\sn_model.c src\sn_pack.c src\sn_pqk.c src\sn_moe.c src\sn_vision.c src\sn_image.c src\sn_gemm.c src\sn_cuda.c %L%\mamba.c %L%\attn.c %L%\pool.c %L%\tokenizer.c %L%\json.c -lm -lpsapi || exit /b 1
gcc %F% -o test_sn_pqk.exe tests\test_sn_pqk.c src\sn_pqk.c src\sn_pack.c -lm || exit /b 1
gcc %F% -o test_sn_spec.exe tests\test_sn_spec.c src\sn_model.c src\sn_pack.c src\sn_pqk.c src\sn_moe.c src\sn_gemm.c src\sn_cuda.c %L%\mamba.c %L%\attn.c %L%\pool.c -lm || exit /b 1
gcc %F% -o test_sn_cache.exe tests\test_sn_cache.c src\sn_model.c src\sn_pack.c src\sn_pqk.c src\sn_moe.c src\sn_gemm.c src\sn_cuda.c %L%\mamba.c %L%\attn.c %L%\pool.c -lm || exit /b 1
gcc %F% -o test_sn_cmdloop.exe tests\test_sn_cmdloop.c || exit /b 1
gcc %F% -o test_sn_pack_manifest.exe tests\test_sn_pack_manifest.c src\sn_pack.c -lm || exit /b 1
echo built




