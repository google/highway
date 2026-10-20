// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hwy/detect_compiler_arch.h"  // HWY_IDE

#ifndef BTREE_KEY_T
#if HWY_IDE
#define BTREE_KEY_T uint32_t
#else
#error "Only include from btreeset_*.cc, which define BTREE_KEY_T"
#endif  // HWY_IDE
#endif  // BTREE_KEY_T

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hwy/base.h"
#include "hwy/contrib/btree/btree_nodes.h"
#include "hwy/contrib/btree/btree_set.h"

#if defined(HIGHWAY_HWY_CONTRIB_BTREE_BTREESET_IMPL_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_BTREE_BTREESET_IMPL_INL_H_
#undef HIGHWAY_HWY_CONTRIB_BTREE_BTREESET_IMPL_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_BTREE_BTREESET_IMPL_INL_H_
#endif

#include "hwy/highway.h"
// After highway.h
#include "hwy/contrib/btree/btree-inl.h"

// Ignore warning that we are defining functions in a header; this is only
// included from btreeset_*.cc.
// NOLINTBEGIN(misc-definitions-in-headers)

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

using TreeEngine = HWY_NAMESPACE::BTree<HWY_NAMESPACE::SetTraits<BTREE_KEY_T>>;
using TreeState = typename TreeEngine::State;
using TreeLeaf = typename TreeEngine::Leaf;

static_assert(
    sizeof(typename TreeEngine::const_iterator) ==
        sizeof(typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator),
    "TreeEngine::const_iterator must match BTreeSet::const_iterator size");
static_assert(
    alignof(typename TreeEngine::const_iterator) ==
        alignof(typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator),
    "TreeEngine::const_iterator must match BTreeSet::const_iterator alignment");

// The *Impl functions below wrap a short-lived engine adapter around the
// BTreeSet's state and run the corresponding engine operation on it. Engine
// copy/move/swap operate on the contents of whatever state the adapter refers
// to, so these transfers need no manual state copying.

void SetClearImpl(TreeState* state) {
  TreeEngine tree(state);
  tree.clear();
}

void SetCopyConstructImpl(TreeState* dst_state, const TreeState* src_state) {
  TreeEngine src(const_cast<TreeState*>(src_state));
  TreeEngine dst(dst_state);
  dst = TreeEngine(src);  // deep copy into a temporary, then move it in
}

void SetCopyAssignImpl(TreeState* dst_state, const TreeState* src_state) {
  TreeEngine src(const_cast<TreeState*>(src_state));
  TreeEngine dst(dst_state);
  dst = src;
}

void SetMoveAssignImpl(TreeState* dst_state, TreeState* src_state) {
  TreeEngine src(src_state);
  TreeEngine dst(dst_state);
  dst = std::move(src);
}

void SetMoveConstructImpl(TreeState* dst_state, TreeState* src_state) {
  // dst_state is a freshly default-initialized (empty) state, so this is
  // the same as move-assignment.
  SetMoveAssignImpl(dst_state, src_state);
}

void SetBuildImpl(const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                  float fill, TreeState* out_state) {
  TreeEngine out(out_state);
  out = TreeEngine::Build(keys, count, fill);
}

bool SetContainsImpl(const TreeState* state, BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  return tree.contains(key);
}

std::pair<const TreeLeaf*, size_t> SetLowerBoundImpl(const TreeState* state,
                                                     BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  auto it = tree.lower_bound(key);
  return {it.leaf(), it.slot()};
}

std::pair<const TreeLeaf*, size_t> SetUpperBoundImpl(const TreeState* state,
                                                     BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  auto it = tree.upper_bound(key);
  return {it.leaf(), it.slot()};
}

std::pair<const TreeLeaf*, size_t> SetFindImpl(const TreeState* state,
                                               BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  auto it = tree.find(key);
  return {it.leaf(), it.slot()};
}

void SetInsertImpl(TreeState* state, BTREE_KEY_T key, TreeLeaf** out_leaf,
                   size_t* out_slot, bool* out_inserted) {
  TreeEngine tree(state);
  auto res = tree.insert(key);
  *out_leaf = res.first.leaf();
  *out_slot = res.first.slot();
  *out_inserted = res.second;
}

void SetEraseImpl(TreeState* state, BTREE_KEY_T key, size_t* out_erased) {
  TreeEngine tree(state);
  *out_erased = tree.erase(key);
}

void SetEraseIterImpl(TreeState* state, const TreeLeaf* leaf, size_t slot,
                      TreeLeaf** out_leaf, size_t* out_slot) {
  TreeEngine tree(state);
  auto it = tree.erase(typename TreeEngine::const_iterator(leaf, slot));
  *out_leaf = it.leaf();
  *out_slot = it.slot();
}

void SetContainsBatchImpl(const TreeState* state,
                          const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                          bool* HWY_RESTRICT out) {
  TreeEngine tree(const_cast<TreeState*>(state));
  tree.ContainsBatch(keys, count, out);
}

void SetFindBatchImpl(const TreeState* state,
                      const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                      void* HWY_RESTRICT out) {
  TreeEngine tree(const_cast<TreeState*>(state));
  tree.FindBatch(keys, count,
                 reinterpret_cast<typename TreeEngine::const_iterator*>(out));
}

void SetLowerBoundBatchImpl(const TreeState* state,
                            const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                            void* HWY_RESTRICT out) {
  TreeEngine tree(const_cast<TreeState*>(state));
  tree.LowerBoundBatch(
      keys, count, reinterpret_cast<typename TreeEngine::const_iterator*>(out));
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREESET_IMPL_INL_H_

#if HWY_ONCE
namespace hwy {

HWY_EXPORT(SetClearImpl);
HWY_EXPORT(SetCopyConstructImpl);
HWY_EXPORT(SetCopyAssignImpl);
HWY_EXPORT(SetMoveConstructImpl);
HWY_EXPORT(SetMoveAssignImpl);
HWY_EXPORT(SetBuildImpl);
HWY_EXPORT(SetContainsImpl);
HWY_EXPORT(SetContainsBatchImpl);
HWY_EXPORT(SetFindBatchImpl);
HWY_EXPORT(SetLowerBoundBatchImpl);
HWY_EXPORT(SetLowerBoundImpl);
HWY_EXPORT(SetUpperBoundImpl);
HWY_EXPORT(SetFindImpl);
HWY_EXPORT(SetInsertImpl);
HWY_EXPORT(SetEraseImpl);
HWY_EXPORT(SetEraseIterImpl);

// Defines the dynamic dispatch entry points for SetDispatch<KeyT>.
// KeyT is strictly uint32_t, int32_t, uint64_t, or int64_t; user-facing
// BTreeSet<KeyT> delegates to these implementations via BitCastScalar and
// pointer reinterpretation.
namespace detail {

void SetDispatch<BTREE_KEY_T>::Clear(State* state) {
  HWY_DYNAMIC_DISPATCH(SetClearImpl)(state);
}

void SetDispatch<BTREE_KEY_T>::CopyConstruct(State* dst, const State* src) {
  HWY_DYNAMIC_DISPATCH(SetCopyConstructImpl)(dst, src);
}

void SetDispatch<BTREE_KEY_T>::CopyAssign(State* dst, const State* src) {
  HWY_DYNAMIC_DISPATCH(SetCopyAssignImpl)(dst, src);
}

void SetDispatch<BTREE_KEY_T>::MoveConstruct(State* dst, State* src) {
  HWY_DYNAMIC_DISPATCH(SetMoveConstructImpl)(dst, src);
}

void SetDispatch<BTREE_KEY_T>::MoveAssign(State* dst, State* src) {
  HWY_DYNAMIC_DISPATCH(SetMoveAssignImpl)(dst, src);
}

void SetDispatch<BTREE_KEY_T>::Build(const BTREE_KEY_T* keys, size_t count,
                                     float fill, State* out_state) {
  HWY_DYNAMIC_DISPATCH(SetBuildImpl)(keys, count, fill, out_state);
}

bool SetDispatch<BTREE_KEY_T>::Contains(const State* state, BTREE_KEY_T key) {
  return HWY_DYNAMIC_DISPATCH(SetContainsImpl)(state, key);
}

std::pair<const typename SetDispatch<BTREE_KEY_T>::Leaf*, size_t>
SetDispatch<BTREE_KEY_T>::LowerBound(const State* state, BTREE_KEY_T key) {
  return HWY_DYNAMIC_DISPATCH(SetLowerBoundImpl)(state, key);
}

std::pair<const typename SetDispatch<BTREE_KEY_T>::Leaf*, size_t>
SetDispatch<BTREE_KEY_T>::UpperBound(const State* state, BTREE_KEY_T key) {
  return HWY_DYNAMIC_DISPATCH(SetUpperBoundImpl)(state, key);
}

std::pair<const typename SetDispatch<BTREE_KEY_T>::Leaf*, size_t>
SetDispatch<BTREE_KEY_T>::Find(const State* state, BTREE_KEY_T key) {
  return HWY_DYNAMIC_DISPATCH(SetFindImpl)(state, key);
}

std::pair<std::pair<typename SetDispatch<BTREE_KEY_T>::Leaf*, size_t>, bool>
SetDispatch<BTREE_KEY_T>::Insert(State* state, BTREE_KEY_T key) {
  Leaf* leaf = nullptr;
  size_t slot = 0;
  bool inserted = false;
  HWY_DYNAMIC_DISPATCH(SetInsertImpl)(state, key, &leaf, &slot, &inserted);
  return {{leaf, slot}, inserted};
}

size_t SetDispatch<BTREE_KEY_T>::Erase(State* state, BTREE_KEY_T key) {
  size_t erased = 0;
  HWY_DYNAMIC_DISPATCH(SetEraseImpl)(state, key, &erased);
  return erased;
}

std::pair<typename SetDispatch<BTREE_KEY_T>::Leaf*, size_t>
SetDispatch<BTREE_KEY_T>::EraseIter(State* state, const Leaf* leaf,
                                    size_t slot) {
  Leaf* out_leaf = nullptr;
  size_t out_slot = 0;
  HWY_DYNAMIC_DISPATCH(SetEraseIterImpl)(state, leaf, slot, &out_leaf,
                                         &out_slot);
  return {out_leaf, out_slot};
}

void SetDispatch<BTREE_KEY_T>::ContainsBatch(const State* state,
                                             const BTREE_KEY_T* keys,
                                             size_t count, bool* out) {
  HWY_DYNAMIC_DISPATCH(SetContainsBatchImpl)(state, keys, count, out);
}

void SetDispatch<BTREE_KEY_T>::FindBatch(const State* state,
                                         const BTREE_KEY_T* keys, size_t count,
                                         void* out) {
  HWY_DYNAMIC_DISPATCH(SetFindBatchImpl)(state, keys, count, out);
}

void SetDispatch<BTREE_KEY_T>::LowerBoundBatch(const State* state,
                                               const BTREE_KEY_T* keys,
                                               size_t count, void* out) {
  HWY_DYNAMIC_DISPATCH(SetLowerBoundBatchImpl)(state, keys, count, out);
}

}  // namespace detail
}  // namespace hwy
#endif  // HWY_ONCE

// NOLINTEND(misc-definitions-in-headers)
