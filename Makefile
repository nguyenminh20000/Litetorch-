CXX ?= g++
CXXFLAGS ?= -std=c++14 -O3 -fPIC -Iinclude

ifeq ($(OS),Windows_NT)
    LDFLAGS ?= -shared -lpthread -lws2_32
    TARGET_LIB := build/liblitetorch.so
else
    LDFLAGS ?= -shared -lpthread -ldl -lrt
    TARGET_LIB := build/liblitetorch.so
endif

SRC_DIR := src
BUILD_DIR := build
OBJ_DIR := $(BUILD_DIR)/objs

SRCS := $(shell find $(SRC_DIR) -name "*.cpp" -not -path "src/bindings/*")
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

.PHONY: all clean

all: $(TARGET_LIB)

$(TARGET_LIB): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
