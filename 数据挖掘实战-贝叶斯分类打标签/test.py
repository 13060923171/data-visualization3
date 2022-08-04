from googletrans import Translator
import re
translator = Translator()
def fy(x):
    try:
        translations = translator.translate(x, dest='en')
        return translations.text
    except:
        return np.NAN

print(fy('Aku lebih setia sama Minecraft'))

def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' ', tweet)
    tweet = re.sub(r'#GTCartoon:', ' ', tweet)
    return tweet

def clean_text(tweet):
    processed_tweet = []
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', ' ', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', ' ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', ' ', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    # 去掉数字
    tweet = re.sub(r'\d+', ' ', tweet)
    # 标点符号
    # tweet = re.sub(r'[^A-Z^a-z^0-9^]', ' ', tweet)
    return tweet
    # processed_tweet.append(tweet)
    # words = tweet.lower().split()
    # words = [w for w in words]
    # for word in words:
    #     word = preprocess_word(word)
    #     # if is_valid_word(word):
    #     processed_tweet.append(word)
    # if len(processed_tweet) != 0:
    #     return ' '.join(processed_tweet)
    # else:
    #     return np.NAN


print(clean_text('Bg要求提供账户'))