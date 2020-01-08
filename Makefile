.DELETE_ON_ERROR:

OBJS := runtime_dispatch.o nanobenchmark.o
TEST_NAMES := arithmetic_test compare_test convert_test hwy_test logical_test memory_test nanobenchmark_test swizzle_test
TESTS := $(foreach i, $(TEST_NAMES), bin/$(i))

CXXFLAGS += -I. -fmerge-all-constants -std=c++17 -O2 \
    -Wno-builtin-macro-redefined -D__DATE__="redacted" \
    -D__TIMESTAMP__="redacted" -D__TIME__="redacted"  \
    -Wall -Wextra -Wformat-security -Wno-unused-function \
    -Wnon-virtual-dtor -Woverloaded-virtual -Wvla

.PHONY: all
all: $(TESTS) benchmark test

.PHONY: clean
clean: ; @rm -rf $(OBJS) bin/ benchmark.o

$(OBJS): %.o: hwy/%.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

benchmark: $(OBJS) hwy/benchmark.cc
	mkdir -p bin && $(CXX) $(CXXFLAGS) $(OBJS) hwy/benchmark.cc -o bin/benchmark

$(TESTS): $(OBJS)
	mkdir -p bin && $(CXX) $(CXXFLAGS) $(subst bin/,hwy/tests/,$@).cc $(OBJS) -o $@

.PHONY: test
test: $(TESTS)
	for name in $^; do echo ---------------------$$name && $$name; done
