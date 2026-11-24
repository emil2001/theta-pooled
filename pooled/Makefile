THETA_DIR ?= $(abspath ..)
CXX ?= g++
CXXFLAGS ?= -O2 -g -Wall -Wextra -fPIC --std=c++17

TARGETS := libtarget_adaptive_mcmc_cov.so libpooled_mcmc_cov.so

.PHONY: all clean

all: $(TARGETS)

libtarget_adaptive_mcmc_cov.so: target_adaptive_mcmc_cov.cpp
	$(CXX) $(CXXFLAGS) -I$(THETA_DIR) -shared -o $@ $< \
		-Wl,-rpath,$(THETA_DIR)/lib -L$(THETA_DIR)/lib -ltheta

libpooled_mcmc_cov.so: pooled_mcmc_cov.cpp
	$(CXX) $(CXXFLAGS) -I$(THETA_DIR) -shared -o $@ $< \
		-Wl,-rpath,$(THETA_DIR)/lib -L$(THETA_DIR)/lib -ltheta

clean:
	$(RM) $(TARGETS)
