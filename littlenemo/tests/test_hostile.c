/* test_hostile.c - loader behavior on hostile fixtures
 * (CODE-REVIEW-2026-11-24.md, findings E1/E3/E4/E5/E6).
 *   test_hostile refuse <dir>   checkpoint open must fail cleanly
 *   test_hostile i32    <dir>   open succeeds, load_f32 must refuse
 */
#include <stdio.h>
#include <string.h>

#include "../src/loader.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: test_hostile <refuse|i32> <dir>\n");
        return 2;
    }
    nemo_checkpoint cp;
    int rc = nemo_checkpoint_open(&cp, argv[2]);
    if (strcmp(argv[1], "refuse") == 0) {
        if (rc == 0) {
            printf("FAIL: hostile checkpoint was accepted\n");
            nemo_checkpoint_close(&cp);
            return 1;
        }
        printf("ok   refused cleanly\n");
        return 0;
    }
    if (strcmp(argv[1], "i32") == 0) {
        if (rc != 0) {
            printf("FAIL: i32 checkpoint should index\n");
            return 1;
        }
        const nemo_tensor_info *t = nemo_find(&cp, "t");
        if (!t) {
            printf("FAIL: tensor t not indexed\n");
            nemo_checkpoint_close(&cp);
            return 1;
        }
        float *v = nemo_load_f32(&cp, t);
        if (v != NULL) {
            printf("FAIL: I32 payload was reinterpreted as float\n");
            nemo_checkpoint_close(&cp);
            return 1;
        }
        printf("ok   I32 load refused on dtype\n");
        nemo_checkpoint_close(&cp);
        return 0;
    }
    printf("unknown mode\n");
    return 2;
}
