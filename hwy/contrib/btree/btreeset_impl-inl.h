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
using TreeState = hwy::BTreeState<BTREE_KEY_T>;

void SetClearImpl(TreeState* state) {
  TreeEngine tree(state);
  tree.clear();
}

void SetCopyConstructImpl(TreeState* dst_state, const TreeState* src_state) {
  TreeEngine src(const_cast<TreeState*>(src_state));
  TreeEngine copy(src);
  *dst_state = *copy.state();
  *copy.state() = TreeState{};
}

void SetCopyAssignImpl(TreeState* dst_state, const TreeState* src_state) {
  TreeEngine src(const_cast<TreeState*>(src_state));
  TreeEngine dst(dst_state);
  dst = src;
}

void SetMoveConstructImpl(TreeState* dst_state, TreeState* src_state) {
  *dst_state = *src_state;
  *src_state = TreeState{};
}

void SetMoveAssignImpl(TreeState* dst_state, TreeState* src_state) {
  TreeEngine dst(dst_state);
  dst.clear();
  *dst_state = *src_state;
  *src_state = TreeState{};
}

void SetBuildImpl(const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                  float fill, TreeState* out_state) {
  auto tree = TreeEngine::Build(keys, count, fill);
  *out_state = *tree.state();
  *tree.state() = TreeState{};
}

bool SetContainsImpl(const TreeState* state, BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  return tree.contains(key);
}

typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator SetLowerBoundImpl(
    const TreeState* state, BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  auto it = tree.lower_bound(key);
  return typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator(
      it.leaf(), it.slot(), state->last_leaf_);
}

typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator SetUpperBoundImpl(
    const TreeState* state, BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  auto it = tree.upper_bound(key);
  return typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator(
      it.leaf(), it.slot(), state->last_leaf_);
}

typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator SetFindImpl(
    const TreeState* state, BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  auto it = tree.find(key);
  return typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator(
      it.leaf(), it.slot(), state->last_leaf_);
}

void SetInsertImpl(TreeState* state, BTREE_KEY_T key,
                   typename hwy::BTreeSet<BTREE_KEY_T>::iterator* out_it,
                   bool* out_inserted) {
  TreeEngine tree(state);
  auto res = tree.insert(key);
  *out_it = typename hwy::BTreeSet<BTREE_KEY_T>::iterator(
      res.first.leaf(), res.first.slot(), state->last_leaf_);
  *out_inserted = res.second;
}

void SetEraseImpl(TreeState* state, BTREE_KEY_T key, size_t* out_erased) {
  TreeEngine tree(state);
  *out_erased = tree.erase(key);
}

void SetContainsBatchImpl(const TreeState* state,
                          const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                          bool* HWY_RESTRICT out) {
  TreeEngine tree(const_cast<TreeState*>(state));
  tree.ContainsBatch(keys, count, out);
}

void SetFindBatchImpl(
    const TreeState* state, const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
    typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator* HWY_RESTRICT out) {
  TreeEngine tree(const_cast<TreeState*>(state));
  static_assert(sizeof(typename TreeEngine::const_iterator) ==
                sizeof(typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator));
  tree.FindBatch(keys, count,
                 reinterpret_cast<typename TreeEngine::const_iterator*>(out));
}

void SetLowerBoundBatchImpl(
    const TreeState* state, const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
    typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator* HWY_RESTRICT out) {
  TreeEngine tree(const_cast<TreeState*>(state));
  static_assert(sizeof(typename TreeEngine::const_iterator) ==
                sizeof(typename hwy::BTreeSet<BTREE_KEY_T>::const_iterator));
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

template <>
void BTreeSet<BTREE_KEY_T>::clear() {
  HWY_DYNAMIC_DISPATCH(SetClearImpl)(&state_);
}

template <>
BTreeSet<BTREE_KEY_T>::BTreeSet(const BTreeSet& other) {
  HWY_DYNAMIC_DISPATCH(SetCopyConstructImpl)(&state_, &other.state_);
}

template <>
BTreeSet<BTREE_KEY_T>& BTreeSet<BTREE_KEY_T>::operator=(const BTreeSet& other) {
  if (this != &other) {
    HWY_DYNAMIC_DISPATCH(SetCopyAssignImpl)(&state_, &other.state_);
  }
  return *this;
}

template <>
BTreeSet<BTREE_KEY_T>::BTreeSet(BTreeSet&& other) noexcept {
  HWY_DYNAMIC_DISPATCH(SetMoveConstructImpl)(&state_, &other.state_);
}

template <>
BTreeSet<BTREE_KEY_T>& BTreeSet<BTREE_KEY_T>::operator=(
    BTreeSet&& other) noexcept {
  if (this != &other) {
    HWY_DYNAMIC_DISPATCH(SetMoveAssignImpl)(&state_, &other.state_);
  }
  return *this;
}

template <>
BTreeSet<BTREE_KEY_T>::~BTreeSet() {
  clear();
}

template <>
BTreeSet<BTREE_KEY_T> BTreeSet<BTREE_KEY_T>::Build(
    const BTREE_KEY_T* sorted_keys, size_t num_keys, float fill_ratio) {
  BTreeSet<BTREE_KEY_T> tree;
  HWY_DYNAMIC_DISPATCH(SetBuildImpl)(sorted_keys, num_keys, fill_ratio,
                                     &tree.state_);
  return tree;
}

template <>
BTreeSet<BTREE_KEY_T>::const_iterator BTreeSet<BTREE_KEY_T>::lower_bound(
    BTREE_KEY_T key) const {
  return HWY_DYNAMIC_DISPATCH(SetLowerBoundImpl)(&state_, key);
}

template <>
BTreeSet<BTREE_KEY_T>::const_iterator BTreeSet<BTREE_KEY_T>::upper_bound(
    BTREE_KEY_T key) const {
  return HWY_DYNAMIC_DISPATCH(SetUpperBoundImpl)(&state_, key);
}

template <>
BTreeSet<BTREE_KEY_T>::const_iterator BTreeSet<BTREE_KEY_T>::find(
    BTREE_KEY_T key) const {
  return HWY_DYNAMIC_DISPATCH(SetFindImpl)(&state_, key);
}

template <>
std::pair<typename BTreeSet<BTREE_KEY_T>::iterator, bool>
BTreeSet<BTREE_KEY_T>::insert(BTREE_KEY_T key) {
  BTreeSet<BTREE_KEY_T>::iterator it;
  bool inserted = false;
  HWY_DYNAMIC_DISPATCH(SetInsertImpl)(&state_, key, &it, &inserted);
  return {it, inserted};
}

template <>
size_t BTreeSet<BTREE_KEY_T>::erase(BTREE_KEY_T key) {
  size_t erased = 0;
  HWY_DYNAMIC_DISPATCH(SetEraseImpl)(&state_, key, &erased);
  return erased;
}

template <>
bool BTreeSet<BTREE_KEY_T>::Contains(BTREE_KEY_T key) const {
  return HWY_DYNAMIC_DISPATCH(SetContainsImpl)(&state_, key);
}

template <>
void BTreeSet<BTREE_KEY_T>::ContainsBatch(const BTREE_KEY_T* keys, size_t count,
                                          bool* out) const {
  HWY_DYNAMIC_DISPATCH(SetContainsBatchImpl)(&state_, keys, count, out);
}

template <>
void BTreeSet<BTREE_KEY_T>::FindBatch(const BTREE_KEY_T* keys, size_t count,
                                      const_iterator* out) const {
  HWY_DYNAMIC_DISPATCH(SetFindBatchImpl)(&state_, keys, count, out);
}

template <>
void BTreeSet<BTREE_KEY_T>::LowerBoundBatch(const BTREE_KEY_T* keys,
                                            size_t count,
                                            const_iterator* out) const {
  HWY_DYNAMIC_DISPATCH(SetLowerBoundBatchImpl)(&state_, keys, count, out);
}

template class HWY_CONTRIB_DLLEXPORT BTreeSet<BTREE_KEY_T>;

}  // namespace hwy
#endif  // HWY_ONCE

// NOLINTEND(misc-definitions-in-headers)
