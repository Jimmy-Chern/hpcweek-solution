/**
 * compress.c
 * * 一个使用 htslib 并行将文件压缩为 BGZF 格式的 C 程序。
 * * 使用 htslib 内置线程池的优化版本 (v1, bgzf_mt)。
 */

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <errno.h>  // 用于错误处理
#include <string.h> // 用于 strerror 函数

// htslib BGZF 功能相关的头文件
#include "htslib/bgzf.h"
// 注意：此版本不包含 htslib/hts.h，因为它依赖较旧的 API

// 缓冲区大小设置为 64 KiB (65536 字节),
// 对应 BGZF 规范中单个未压缩块的最大大小。
const int BUF_SIZE = 65536;

// 要使用的压缩线程数。
// 测评机有 52 个核心。我们使用 51 个线程进行压缩，
// 留下主线程 (第52个核心) 来处理 I/O (读取)。
const int N_THREADS = 51;

int main(int argc, char **argv)
{
    // 校验命令行参数
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <infile> <outfile>\n", argv[0]);
        return 1;
    }

    const char *infile = argv[1];
    const char *outfile = argv[2];

    // 打开输入文件
    FILE *fin = fopen(infile, "rb");
    if (fin == NULL) {
        fprintf(stderr, "错误: 无法打开输入文件 '%s': %s\n", infile, strerror(errno));
        return 1;
    }

    // 以 BGZF 格式打开输出文件 (写入模式 "w")
    BGZF *out = bgzf_open(outfile, "w");
    if (out == NULL) {
        fprintf(stderr, "错误: 无法打开 BGZF 输出文件 '%s'\n", outfile);
        fclose(fin);
        return 1;
    }

    // ===================================================================
    // 优化：激活多线程 (使用较旧的 bgzf_mt API)
    // 通知 htslib 使用一个包含 N_THREADS 个线程的线程池。
    // 主线程将负责读取数据块并提交到这个线程池。
    // 256 是队列大小（可以提交的待处理块的数量）
    if (bgzf_mt(out, N_THREADS, 256) < 0) {
        fprintf(stderr, "错误: 设置 BGZF 多线程时出错\n");
        bgzf_close(out);
        fclose(fin);
        return 1;
    }
    // ===================================================================

    // 分配读取缓冲区
    uint8_t *buf = (uint8_t *)calloc(BUF_SIZE, sizeof(uint8_t));
    if (buf == NULL) {
        fprintf(stderr, "错误: 为缓冲区分配内存时出错\n");
        bgzf_close(out);
        fclose(fin);
        return 1;
    }

    size_t nread;
    int ret_code = 0; // 跟踪退出状态

    // 主循环：以 64 KiB 的块大小读取数据
    while ((nread = fread(buf, 1, BUF_SIZE, fin)) > 0)
    {
        // 将读取到的数据块提交给 BGZF 写入器。
        // 线程池中的线程会自动并行执行压缩操作。
        // bgzf_write 在失败时返回 -1
        int ret = bgzf_write(out, (const uint8_t *)buf, (int)nread);
        
        // 检查 bgzf_write 是否成功写入
        if (ret < 0) // 仅当 ret == -1 (失败) 时为 true
        {
            fprintf(stderr, "错误: 写入 BGZF 文件时出错\n");
            ret_code = 1;
            break; // 遇到写入错误，退出循环，但继续执行清理步骤
        }
    }

    // 检查 fread() 是否因为出错 (而不是到达文件末尾 EOF) 而终止
    if (ferror(fin)) {
        fprintf(stderr, "错误: 从输入文件 '%s' 读取时出错\n", infile);
        ret_code = 1;
    }

    // 关闭 BGZF 文件。
    // bgzf_close() 会在内部调用 bgzf_flush() 来确保
    // 所有的缓冲数据块 (包括最后一块) 都被写入，
    // 然后等待所有工作线程结束, 最后关闭文件。
    if (bgzf_close(out) < 0)
    {
        fprintf(stderr, "错误: 关闭 BGZF 文件时出错\n");
        ret_code = 1;
    }
    
    // 关闭输入文件
    if (fclose(fin) != 0) {
        perror("错误: 关闭输入文件时出错");
        ret_code = 1;
    }

    // 修复了内存泄漏：释放缓冲区
    free(buf); 

    return ret_code;
}


