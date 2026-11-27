@echo off
rem SuperNemo pieces (pure C99, no locks). Run from this folder.
setlocal
set F=-O2 -std=c99 -Wall -Wextra -D_FILE_OFFSET_BITS=64 -D_POSIX_C_SOURCE=200809L
set L=..\littlenemo\src
gcc %F% -o test_sn_moe.exe tests\test_sn_moe.c src\sn_moe.c -lm || exit /b 1
gcc %F% -o sn_packcheck.exe tests\sn_packcheck.c src\sn_pack.c -lm || exit /b 1
gcc %F% -o sn_preflight.exe tests\sn_preflight.c src\sn_pack.c -lm || exit /b 1
gcc %F% -o sn_mktiny.exe tests\sn_mktiny.c -lm || exit /b 1
gcc %F% -o test_sn_model.exe tests\test_sn_model.c src\sn_model.c src\sn_pack.c src\sn_moe.c %L%\mamba.c %L%\attn.c %L%\pool.c -lm || exit /b 1
gcc %F% -o sn-engine.exe src\sn_engine.c src\sn_model.c src\sn_pack.c src\sn_moe.c %L%\mamba.c %L%\attn.c %L%\pool.c %L%\tokenizer.c %L%\json.c -lm -lpsapi || exit /b 1
echo built
