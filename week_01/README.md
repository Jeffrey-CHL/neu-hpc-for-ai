# Week 01 Assignment

## Part 1. Feel the Magic 
In this section, I successfully compiled and ran Karpathy’s **llama2.c** project with a small 15M TinyStories model.

### Steps
```bash
# clone the repo
git clone https://github.com/karpathy/llama2.c.git
cd llama2.c

# download the small TinyStories 15M model
curl -L https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin -o stories15M.bin

# compile and run
make run
./run stories15M.bin

Output

The program generated a short story and reported the token generation speed:

Once upon a time, there was a little girl named Lily. ...
achieved tok/s: 110.907424

Then I tried the optimized build:

make runfast
./run stories15M.bin

And the speed improved significantly:

Once upon a time, there was a little boy named Tim. ...
achieved tok/s: 673.553719

➡️ [Insert screenshot of terminal output here]

⸻

Part 2. HuggingFace Access

I logged in with huggingface_hub CLI and confirmed I could download LLaMA-2-7B weights to my local machine.

Steps:

pip install huggingface_hub
hf auth login   # logged in with my token
hf download meta-llama/Llama-2-7b-hf --local-dir ./llama2-7b

Download completed successfully, which confirms that my access request for LLaMA 2 models has been approved.

⸻

Part 3. Reflection
	•	This week’s assignment helped me understand how C code can be used to run a simplified Transformer model.
	•	I experienced the trade-off between -O3 vs -Ofast optimizations and how compilation flags can significantly improve runtime speed.
	•	I also learned how to use HuggingFace CLI to authenticate and download large pretrained models.


