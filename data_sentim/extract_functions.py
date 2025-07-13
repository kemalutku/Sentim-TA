import os, time

log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def extract_titles(model, tokenizer, PAD_ID, headlines):
    number_of_headlines = len(headlines)
    prompt = ("You are a news editor on a tv show.\n"
              f"You are asked to give one word summaries for {number_of_headlines} headlines:\n"
              f"Headlines: \n{chr(10).join(headlines)}\n---\n"
              "Give a impactful topic of each headline with only 1 word. Write down the topics in this format:\n"
              "1-{topic}\n2-{topic}\n3-{topic}\n...\n"
              "Do not say anything else, do not include a note. If you say anything other than the format you lose."
              "Just say the topic in this format 1-{topic}\n")
    toks = tokenizer(prompt, return_tensors="pt").to("cuda")
    gen = model.generate(**toks, max_new_tokens=1024, pad_token_id=PAD_ID, do_sample=False)
    return tokenizer.decode(gen[0][toks["input_ids"].shape[1]:], skip_special_tokens=True).strip(" .\n")
