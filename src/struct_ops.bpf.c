/* SPDX-License-Identifier: GPL-2.0 */
#include <vmlinux.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include "module/bpf_testmod.h"

char _license[] SEC("license") = "GPL";

/* Declare the external kfunc from nvidia-uvm module */
extern int bpf_uvm_strstr(const char *str, u32 str__sz, const char *substr, u32 substr__sz) __ksym;

/* Implement the struct_ops callbacks */
SEC("struct_ops/uvm_bpf_test_trigger_kfunc")
int BPF_PROG(uvm_bpf_test_trigger_kfunc, const char *buf, int len)
{
	char read_buf[64] = {0};
	int read_len = len < sizeof(read_buf) ? len : sizeof(read_buf) - 1;
	char str[] = "Hello, GPU world!";
	char substr[] = "GPU";
	int result;

	bpf_printk("BPF uvm_bpf_test_trigger_kfunc called with buffer length %d\n", len);

	/* Test the kfunc */
	result = bpf_uvm_strstr(str, sizeof(str) - 1, substr, sizeof(substr) - 1);
	if (result != -1) {
		bpf_printk("'%s' found in '%s' at index %d\n", substr, str, result);
	} else {
		bpf_printk("'%s' not found in '%s'\n", substr, str);
	}

	/* Safely read from kernel buffer using bpf_probe_read_kernel */
	if (buf && read_len > 0) {
		long ret = bpf_probe_read_kernel(read_buf, read_len, buf);
		if (ret == 0) {
			/* Successfully read buffer - print first few characters */
			bpf_printk("Buffer content: '%c%c%c%c'\n",
				   read_buf[0], read_buf[1], read_buf[2], read_buf[3]);
			bpf_printk("Full buffer: %s\n", read_buf);
		} else {
			bpf_printk("Failed to read buffer, ret=%ld\n", ret);
		}
	}

	return len;
}

/* Define the struct_ops map */
SEC(".struct_ops")
struct uvm_gpu_ext uvm_ops = {
	.uvm_bpf_test_trigger_kfunc = (void *)uvm_bpf_test_trigger_kfunc,
};
