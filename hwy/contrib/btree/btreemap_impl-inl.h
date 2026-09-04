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
#error "Only include from btreemap_*.cc, which define BTREE_KEY_T"
#endif  // HWY_IDE
#endif  // BTREE_KEY_T

#ifndef BTREE_VALUE_T
#if HWY_IDE
#define BTREE_VALUE_T uint32_t
#else
#error "Only include from btreemap_*.cc, which define BTREE_VALUE_T"
#endif  // HWY_IDE
#endif  // BTREE_VALUE_T

#include <cstddef>
#include <cstdint>
#include <utility>

#include "hwy/base.h"
#include "hwy/contrib/btree/btree_map.h"
#include "hwy/contrib/btree/btree_nodes.h"

#if defined(HIGHWAY_HWY_CONTRIB_BTREE_BTREEMAP_IMPL_INL_H_) == \
    defined(HWY_TARGET_TOGGLE)
#ifdef HIGHWAY_HWY_CONTRIB_BTREE_BTREEMAP_IMPL_INL_H_
#undef HIGHWAY_HWY_CONTRIB_BTREE_BTREEMAP_IMPL_INL_H_
#else
#define HIGHWAY_HWY_CONTRIB_BTREE_BTREEMAP_IMPL_INL_H_
#endif

#include "hwy/highway.h"
// After highway.h
#include "hwy/contrib/btree/btree-inl.h"

// Ignore warning that we are defining functions in a header; this is only
// included from btreemap_*.cc.
// NOLINTBEGIN(misc-definitions-in-headers)

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

using TreeEngine =
    HWY_NAMESPACE::BTree<HWY_NAMESPACE::MapTraits<BTREE_KEY_T, BTREE_VALUE_T>>;
using TreeState = typename TreeEngine::State;
using TreeLeaf = typename TreeEngine::Leaf;

static_assert(
    sizeof(typename TreeEngine::const_iterator) ==
        sizeof(
            typename hwy::BTreeMap<BTREE_KEY_T, BTREE_VALUE_T>::const_iterator),
    "TreeEngine::const_iterator must match BTreeMap::const_iterator size");
static_assert(
    alignof(typename TreeEngine::const_iterator) ==
        alignof(
            typename hwy::BTreeMap<BTREE_KEY_T, BTREE_VALUE_T>::const_iterator),
    "TreeEngine::const_iterator must match BTreeMap::const_iterator alignment");

void MapClearImpl(TreeState* state) {
  TreeEngine tree(state);
  tree.clear();
}

void MapCopyConstructImpl(TreeState* dst_state, const TreeState* src_state) {
  TreeEngine src(const_cast<TreeState*>(src_state));
  TreeEngine copy(src);
  *dst_state = *copy.state();
  *copy.state() = TreeState{};
}

void MapCopyAssignImpl(TreeState* dst_state, const TreeState* src_state) {
  TreeEngine src(const_cast<TreeState*>(src_state));
  TreeEngine dst(dst_state);
  dst = src;
}

void MapMoveConstructImpl(TreeState* dst_state, TreeState* src_state) {
  *dst_state = *src_state;
  *src_state = TreeState{};
}

void MapMoveAssignImpl(TreeState* dst_state, TreeState* src_state) {
  TreeEngine dst(dst_state);
  dst.clear();
  *dst_state = *src_state;
  *src_state = TreeState{};
}

void MapBuildImpl(const BTREE_KEY_T* HWY_RESTRICT keys,
                  const void* HWY_RESTRICT values, size_t count, float fill,
                  TreeState* out_state) {
  auto tree = TreeEngine::Build(keys, static_cast<const BTREE_VALUE_T*>(values),
                                count, fill);
  *out_state = *tree.state();
  *tree.state() = TreeState{};
}

bool MapContainsImpl(const TreeState* state, BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  return tree.contains(key);
}

std::pair<const TreeLeaf*, size_t> MapLowerBoundImpl(const TreeState* state,
                                                     BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  auto it = tree.lower_bound(key);
  return {it.leaf(), it.slot()};
}

std::pair<const TreeLeaf*, size_t> MapUpperBoundImpl(const TreeState* state,
                                                     BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  auto it = tree.upper_bound(key);
  return {it.leaf(), it.slot()};
}

std::pair<const TreeLeaf*, size_t> MapFindImpl(const TreeState* state,
                                               BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  auto it = tree.find(key);
  return {it.leaf(), it.slot()};
}

const BTREE_VALUE_T* MapFindValueImpl(const TreeState* state, BTREE_KEY_T key) {
  TreeEngine tree(const_cast<TreeState*>(state));
  return tree.FindValue(key);
}

void MapInsertImpl(TreeState* state, BTREE_KEY_T key,
                   const BTREE_VALUE_T& value, bool assign_if_exists,
                   TreeLeaf** out_leaf, size_t* out_slot, bool* out_inserted) {
  TreeEngine tree(state);
  auto res = assign_if_exists ? tree.insert_or_assign(key, value)
                              : tree.insert(key, value);
  *out_leaf = res.first.leaf();
  *out_slot = res.first.slot();
  *out_inserted = res.second;
}

void MapEraseImpl(TreeState* state, BTREE_KEY_T key, size_t* out_erased) {
  TreeEngine tree(state);
  *out_erased = tree.erase(key);
}

void MapContainsBatchImpl(const TreeState* state,
                          const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                          bool* HWY_RESTRICT out) {
  TreeEngine tree(const_cast<TreeState*>(state));
  tree.ContainsBatch(keys, count, out);
}

void MapFindBatchImpl(const TreeState* state,
                      const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                      void* HWY_RESTRICT out) {
  TreeEngine tree(const_cast<TreeState*>(state));
  tree.FindBatch(keys, count,
                 reinterpret_cast<typename TreeEngine::const_iterator*>(out));
}

void MapLowerBoundBatchImpl(const TreeState* state,
                            const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                            void* HWY_RESTRICT out) {
  TreeEngine tree(const_cast<TreeState*>(state));
  tree.LowerBoundBatch(
      keys, count, reinterpret_cast<typename TreeEngine::const_iterator*>(out));
}

void MapLookupBatchImpl(const TreeState* state,
                        const BTREE_KEY_T* HWY_RESTRICT keys, size_t count,
                        void* HWY_RESTRICT out_values,
                        bool* HWY_RESTRICT out_found) {
  TreeEngine tree(const_cast<TreeState*>(state));
  tree.LookupBatch(keys, count, static_cast<BTREE_VALUE_T*>(out_values),
                   out_found);
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_BTREE_BTREEMAP_IMPL_INL_H_

#if HWY_ONCE
namespace hwy {

HWY_EXPORT(MapClearImpl);
HWY_EXPORT(MapCopyConstructImpl);
HWY_EXPORT(MapCopyAssignImpl);
HWY_EXPORT(MapMoveConstructImpl);
HWY_EXPORT(MapMoveAssignImpl);
HWY_EXPORT(MapBuildImpl);
HWY_EXPORT(MapContainsImpl);
HWY_EXPORT(MapContainsBatchImpl);
HWY_EXPORT(MapFindBatchImpl);
HWY_EXPORT(MapLowerBoundBatchImpl);
HWY_EXPORT(MapLookupBatchImpl);
HWY_EXPORT(MapLowerBoundImpl);
HWY_EXPORT(MapUpperBoundImpl);
HWY_EXPORT(MapFindImpl);
HWY_EXPORT(MapFindValueImpl);
HWY_EXPORT(MapInsertImpl);
HWY_EXPORT(MapEraseImpl);

// Defines the dynamic dispatch entry points for MapDispatch<KeyT,
// StorageValueT>. StorageValueT is strictly uint32_t or uint64_t; user-facing
// BTreeMap<KeyT, ValueT> delegates to these implementations via BitCastScalar
// and pointer reinterpretation.
namespace detail {

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Clear(State* state) {
  HWY_DYNAMIC_DISPATCH(MapClearImpl)(state);
}

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::CopyConstruct(State* dst,
                                                            const State* src) {
  HWY_DYNAMIC_DISPATCH(MapCopyConstructImpl)(dst, src);
}

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::CopyAssign(State* dst,
                                                         const State* src) {
  HWY_DYNAMIC_DISPATCH(MapCopyAssignImpl)(dst, src);
}

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::MoveConstruct(State* dst,
                                                            State* src) {
  HWY_DYNAMIC_DISPATCH(MapMoveConstructImpl)(dst, src);
}

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::MoveAssign(State* dst,
                                                         State* src) {
  HWY_DYNAMIC_DISPATCH(MapMoveAssignImpl)(dst, src);
}

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Build(const BTREE_KEY_T* keys,
                                                    const void* values,
                                                    size_t count, float fill,
                                                    State* out_state) {
  HWY_DYNAMIC_DISPATCH(MapBuildImpl)(keys, values, count, fill, out_state);
}

bool MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Contains(const State* state,
                                                       BTREE_KEY_T key) {
  return HWY_DYNAMIC_DISPATCH(MapContainsImpl)(state, key);
}

const BTREE_VALUE_T* MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::FindValue(
    const State* state, BTREE_KEY_T key) {
  return HWY_DYNAMIC_DISPATCH(MapFindValueImpl)(state, key);
}

std::pair<const typename MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Leaf*, size_t>
MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::LowerBound(const State* state,
                                                    BTREE_KEY_T key) {
  return HWY_DYNAMIC_DISPATCH(MapLowerBoundImpl)(state, key);
}

std::pair<const typename MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Leaf*, size_t>
MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::UpperBound(const State* state,
                                                    BTREE_KEY_T key) {
  return HWY_DYNAMIC_DISPATCH(MapUpperBoundImpl)(state, key);
}

std::pair<const typename MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Leaf*, size_t>
MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Find(const State* state,
                                              BTREE_KEY_T key) {
  return HWY_DYNAMIC_DISPATCH(MapFindImpl)(state, key);
}

std::pair<
    std::pair<typename MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Leaf*, size_t>,
    bool>
MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Insert(State* state, BTREE_KEY_T key,
                                                BTREE_VALUE_T value,
                                                bool assign_if_exists) {
  Leaf* leaf = nullptr;
  size_t slot = 0;
  bool inserted = false;
  HWY_DYNAMIC_DISPATCH(MapInsertImpl)(state, key, value, assign_if_exists,
                                      &leaf, &slot, &inserted);
  return {{leaf, slot}, inserted};
}

size_t MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::Erase(State* state,
                                                      BTREE_KEY_T key) {
  size_t erased = 0;
  HWY_DYNAMIC_DISPATCH(MapEraseImpl)(state, key, &erased);
  return erased;
}

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::ContainsBatch(
    const State* state, const BTREE_KEY_T* keys, size_t count, bool* out) {
  HWY_DYNAMIC_DISPATCH(MapContainsBatchImpl)(state, keys, count, out);
}

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::FindBatch(const State* state,
                                                        const BTREE_KEY_T* keys,
                                                        size_t count,
                                                        void* out) {
  HWY_DYNAMIC_DISPATCH(MapFindBatchImpl)(state, keys, count, out);
}

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::LowerBoundBatch(
    const State* state, const BTREE_KEY_T* keys, size_t count, void* out) {
  HWY_DYNAMIC_DISPATCH(MapLowerBoundBatchImpl)(state, keys, count, out);
}

void MapDispatch<BTREE_KEY_T, BTREE_VALUE_T>::LookupBatch(
    const State* state, const BTREE_KEY_T* keys, size_t count, void* out_values,
    bool* out_found) {
  HWY_DYNAMIC_DISPATCH(MapLookupBatchImpl)(state, keys, count, out_values,
                                           out_found);
}

}  // namespace detail
}  // namespace hwy
#endif  // HWY_ONCE

// NOLINTEND(misc-definitions-in-headers)
