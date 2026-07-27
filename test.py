from transformers import AutoModelForCausalLM, AutoTokenizer

# 写完整绝对路径
model_path = r"C:\Users\marym\Qwen2.5-7B-Instruct"
print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("加载模型权重...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="cpu"
)
print("🎉 模型加载成功！")