/* SPDX-License-Identifier: GPL-2.0 */
/*
 * nvidia_rm_types.h - NVIDIA RM (Resource Manager) type definitions for eBPF
 *
 * These types are extracted from open-gpu-kernel-modules source code.
 * Used to hook into NVIDIA driver functions for GPU scheduling control.
 *
 * Key functions to hook:
 * - kchangrpapiConstruct_IMPL: TSG creation, get hClient/hTsg handles
 * - kchangrpapiDestruct_IMPL:  TSG destruction
 *
 * Reference: src/nvidia/inc/libraries/resserv/rs_resource.h
 *            src/nvidia/generated/g_kernel_channel_group_api_nvoc.h
 */

#ifndef __NVIDIA_RM_TYPES_H__
#define __NVIDIA_RM_TYPES_H__

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute push (__attribute__((preserve_access_index)), apply_to = record)
#endif

/* Basic NVIDIA types */
typedef unsigned char NvU8;
typedef unsigned short NvU16;
typedef unsigned int NvU32;
typedef unsigned long long NvU64;
typedef int NvS32;
typedef long long NvS64;
typedef NvU32 NvHandle;
typedef NvU32 NvV32;
typedef unsigned char NvBool;

#define NV_TRUE  1
#define NV_FALSE 0
#define NV01_NULL_OBJECT 0

/*
 * RS_RES_ALLOC_PARAMS_INTERNAL - Resource allocation parameters
 *
 * This is the third argument to kchangrpapiConstruct_IMPL.
 * Contains hClient, hParent, hResource which we need for preempt.
 *
 * From: src/nvidia/inc/libraries/resserv/rs_resource.h:78
 */
struct RS_RES_ALLOC_PARAMS_INTERNAL {
    NvHandle hClient;       /* [in] The handle of the resource's client */
    NvHandle hParent;       /* [in] The handle of the resource's parent */
    NvHandle hResource;     /* [inout] Server assigns handle if 0, or uses provided value */
    NvU32 externalClassId;  /* [in] External class ID of resource */
    NvHandle hDomain;       /* UNUSED */

    /* Internal use only - we don't need these for eBPF tracing */
    void *pLockInfo;        /* RS_LOCK_INFO* */
    void *pClient;          /* RsClient* */
    void *pResourceRef;     /* RsResourceRef* */
    NvU32 allocFlags;
    NvU32 allocState;
    void *pSecInfo;         /* API_SECURITY_INFO* */

    void *pAllocParams;     /* [in] Copied-in allocation parameters */
    NvU32 paramsSize;       /* [in] Copied-in allocation parameters size */

    /* Dupe alloc fields */
    void *pSrcClient;       /* RsClient* */
    void *pSrcRef;          /* RsResourceRef* */

    void *pRightsRequested; /* RS_ACCESS_MASK* */
    /* RS_ACCESS_MASK rightsRequestedCopy; - skip for simplicity */
    /* void *pRightsRequired; - skip for simplicity */
};

typedef struct RS_RES_ALLOC_PARAMS_INTERNAL RS_RES_ALLOC_PARAMS_INTERNAL;

/*
 * CALL_CONTEXT - Second argument to kchangrpapiConstruct_IMPL
 *
 * We don't need to read this structure, but need to know it exists
 * for function signature purposes.
 */
struct CALL_CONTEXT {
    void *pResourceRef;     /* RsResourceRef* */
    void *pClient;          /* RsClient* */
    void *pServer;          /* RsServer* */
    void *pLockInfo;        /* RS_LOCK_INFO* */
    void *pControlParams;
    void *pContextRef;
    NvU32 secInfo;
    /* ... more fields we don't need */
};

typedef struct CALL_CONTEXT CALL_CONTEXT;

/*
 * KernelChannelGroupApi - First argument to kchangrpapiConstruct_IMPL
 *
 * This is a complex NVOC object. We only need minimal fields.
 * The actual structure is much larger with vtables etc.
 *
 * From: src/nvidia/generated/g_kernel_channel_group_api_nvoc.h
 */
struct KernelChannelGroupApi {
    /* NVOC base classes and metadata - skip these */
    char _nvoc_padding[0x100];  /* Approximate offset to fields we care about */

    /* Fields we might want to access */
    void *pKernelChannelGroup;  /* KernelChannelGroup* */
    NvHandle hLegacykCtxShareSync;
    NvHandle hLegacykCtxShareAsync;
    NvHandle hVASpace;
    /* GPreempt patch adds: NvU64 threadId; */
};

typedef struct KernelChannelGroupApi KernelChannelGroupApi;

/*
 * KernelChannelGroup - The actual TSG object
 *
 * From: src/nvidia/generated/g_kernel_channel_group_nvoc.h
 */
struct KernelChannelGroup {
    char _nvoc_padding[0x100];  /* Skip NVOC metadata */

    /* Key fields - offsets need to be determined empirically */
    NvU32 grpID;                /* TSG ID */
    void *pChanList;            /* Channel list */
    NvU32 chanCount;            /* Number of channels */
    NvU64 timeslice;            /* Timeslice in us */
    NvU32 interleaveLevel;      /* Priority level */
    NvU32 runlistId;            /* Runlist this TSG belongs to */
};

typedef struct KernelChannelGroup KernelChannelGroup;

/*
 * NVOS54_PARAMETERS - RM Control ioctl parameters
 *
 * Used for NV_ESC_RM_CONTROL ioctl.
 * From: src/common/sdk/nvidia/inc/nvos.h
 */
struct NVOS54_PARAMETERS {
    NvHandle hClient;
    NvHandle hObject;
    NvV32 cmd;
    NvU32 flags;
    void *params;        /* NvP64 - 64-bit pointer */
    NvU32 paramsSize;
    NvV32 status;
} __attribute__((packed, aligned(8)));

typedef struct NVOS54_PARAMETERS NVOS54_PARAMETERS;

/*
 * NVA06C_CTRL_PREEMPT_PARAMS - Preempt control parameters
 *
 * From: src/common/sdk/nvidia/inc/ctrl/ctrla06c.h
 */
struct NVA06C_CTRL_PREEMPT_PARAMS {
    NvBool bWait;           /* Wait for preempt completion */
    NvBool bManualTimeout;  /* Use custom timeout */
    NvU32 timeoutUs;        /* Timeout in microseconds */
};

typedef struct NVA06C_CTRL_PREEMPT_PARAMS NVA06C_CTRL_PREEMPT_PARAMS;

/*
 * NVA06C_CTRL_TIMESLICE_PARAMS - Timeslice control parameters
 */
struct NVA06C_CTRL_TIMESLICE_PARAMS {
    NvU64 timesliceUs;
};

typedef struct NVA06C_CTRL_TIMESLICE_PARAMS NVA06C_CTRL_TIMESLICE_PARAMS;

/*
 * NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS - Interleave level control
 */
struct NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS {
    NvU32 tsgInterleaveLevel;
};

typedef struct NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS NVA06C_CTRL_INTERLEAVE_LEVEL_PARAMS;

/* NVIDIA escape codes for ioctl */
#define NV_ESC_RM_CONTROL   0x2A
#define NV_ESC_RM_ALLOC     0x2B

/* Control command IDs */
#define NVA06C_CTRL_CMD_PREEMPT             0xa06c0105
#define NVA06C_CTRL_CMD_SET_TIMESLICE       0xa06c0103
#define NVA06C_CTRL_CMD_GET_TIMESLICE       0xa06c0104
#define NVA06C_CTRL_CMD_GET_INFO            0xa06c0106
#define NVA06C_CTRL_CMD_SET_INTERLEAVE_LEVEL 0xa06c0107
#define NVA06C_CTRL_CMD_GPFIFO_SCHEDULE     0xa06c0101

/* Interleave levels (priority) */
#define NVA06C_CTRL_INTERLEAVE_LEVEL_LOW    0
#define NVA06C_CTRL_INTERLEAVE_LEVEL_MEDIUM 1
#define NVA06C_CTRL_INTERLEAVE_LEVEL_HIGH   2

/* Max timeout for preempt in microseconds */
#define NVA06C_CTRL_CMD_PREEMPT_MAX_MANUAL_TIMEOUT_US 1000000

/* Channel class IDs - for identifying TSG creation */
#define AMPERE_CHANNEL_GPFIFO_A   0xc56f
#define KEPLER_CHANNEL_GROUP_A    0xa06c

#ifndef BPF_NO_PRESERVE_ACCESS_INDEX
#pragma clang attribute pop
#endif

#endif /* __NVIDIA_RM_TYPES_H__ */
