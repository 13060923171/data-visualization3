from maha.cleaners.functions import normalize
sample_text = "أنا وأخي علي في المكتبةِ نَطلعُ على موضوع البرمجه"
print(normalize(sample_text, alef=False, all=True))
