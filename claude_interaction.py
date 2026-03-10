from litai import LLM

# initialize the llm with a claude model
# you can choose different claude models available through litai, e.g.,
# "anthropic/claude-3-haiku-20240307" or "anthropic/claude-3-sonnet-20240229"
llm = LLM(model="anthropic/claude-3-haiku-20240307")

# your prompt for claude
prompt = "tell me a short, interesting fact about lightning."

print(f"sending prompt to claude: '{prompt}'")

# get the answer from claude
answer = llm.chat(prompt)

print("\nclaude's response:")
print(answer)