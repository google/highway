# Highway Prefetch Pipelining (`//hwy/contrib/pipeline`)

<!--*
# Document freshness: For more information, see go/fresh-source.
freshness: { owner: 'qhliao' reviewed: '2026-06-10' }
*-->

## Overview

This directory provides abstractions for **Dual-Tier Software Prefetching**,
designed to maximize execution throughput on workloads bound by memory latency
and hardware prefetcher limitations (such as L1 Line Fill Buffer starvation).

By decoupling memory traversals into two separate hardware horizons (a Deep L3
fetch and a Shallow L1 fetch), pipelines can hide massive DRAM latencies while
safely avoiding cache churn and buffer exhaustion.

## Status

**Experimental / Alpha**

The dual tier prefetch loop abstractions are actively in development. While the
core algorithms deliver significant performance enhancements on memory-bound
workloads, the **API is subject to iteration and change**.

Users should expect potential breaking changes in parameter structures, callback
signatures in future releases.
