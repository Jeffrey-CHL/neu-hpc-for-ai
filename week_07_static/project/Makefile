    BIN = bin
    SRC = src/flash_attn.cu
    OUT = $(BIN)/flash_attn

    NVCC = nvcc
    NVFLAGS = -O3 -arch=sm_80

    .PHONY: all clean

    all: $(OUT)

    $(OUT): $(SRC) | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ $<

    $(BIN):
	mkdir -p $(BIN)

    clean:
	rm -rf $(BIN)
