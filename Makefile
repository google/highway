.DELETE_ON_ERROR:

OBJS := runtime_dispatch.o
TEST_NAMES := arithmetic_test compare_test convert_test hwy_test logical_test memory_test swizzle_test
TESTS := $(foreach i, $(TEST_NAMES), bin/$(i))

CXXFLAGS += -I. -fmerge-all-constants \
    -Wno-builtin-macro-redefined -D__DATE__="redacted" \
    -D__TIMESTAMP__="redacted" -D__TIME__="redacted"  \
    -Wall -Wextra -Wformat-security -Wno-unused-function \
    -Wnon-virtual-dtor -Woverloaded-virtual -Wvla

.PHONY: all
all: $(TESTS) test

.PHONY: clean
clean: ; @rm -rf $(OBJS) bin/

$(OBJS): %.o: hwy/%.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(TESTS): $(OBJS)
	mkdir -p bin && $(CXX) $(CXXFLAGS) $(subst bin/,hwy/tests/,$@).cc $(OBJS) -o $@

.PHONY: test
test: $(TESTS)
	for name in $^; do echo ---------------------$$name && $$name; done
