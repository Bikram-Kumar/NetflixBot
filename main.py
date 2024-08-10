# %% [markdown]
# # Netflix Bot
# This ChatBot answers queries related to Netflic movies and shows. It uses `MultinomialNB`and `Natural Language Processing` techniques to train the bot, understand user queries and respond to them.

# %%
import numpy as np
import nltk
import re

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk import regexp_tokenize


# %% [markdown]
# ### Loading the dataset and processing it

# %%
movie_list_file = open("dataset/netflix_titles.csv", mode='r', encoding='utf8')

header = movie_list_file.readline().split(',')

# number of movies to load from file... this may impact performance (total is around 8800)
MOVIES_LEN = 2000

movie_list = movie_list_file.readlines()[:MOVIES_LEN]

movie_list_file.close()

HEADER_LEN = 11
field_indices = {}

for i in range(HEADER_LEN):
    field_indices[header[i]] = i
    
field_indices["description"] = 11

print(field_indices)
# print(*movie_list, sep="\n")                      # uncomment to see movies list


# %% [markdown]
# ### Vectorize the text and extract TF-IDF

# %%
# the regex '\w+' selects all words including single character words (like 'I', 'a', '2') which are ignored by the deafult regex
vectorizer = TfidfVectorizer(token_pattern=r'\w+')

data_transformed = vectorizer.fit_transform(movie_list)


# %% [markdown]
# ### Train

# %%

clf = MultinomialNB()
# use movie index as the target value
clf.fit(data_transformed, range(MOVIES_LEN))


# %% [markdown]
# #### Predict what the user query asks for

# %%

def get_movie_index(query):
    return clf.predict(vectorizer.transform([query]))[0]

# %% [markdown]
# #### Getting movie details
# The predicted index is then used to get the details of movies from processed data. The function `get_movie_detail` extracts the asked fields by the user from the CSV data, and generates a human friendly response containing the details of the movie.

# %%


def get_movie_detail(index, fields=[]):

    # the csv contains commas inside some fields (like cast), so split at commas with no following whitespace
    movie_data = re.split(r',(?=\S)', movie_list[index])

    fi = field_indices                     # declare an alias for better look

    movie_type = movie_data[fi["type"]]
    movie_title = movie_data[fi["title"]]
    release_year = movie_data[fi["release_year"]]

    movie_detail = f"\"{movie_title}\" is a {movie_type} released in {release_year} on Netflix. "

    if len(fields) > 1:
        movie_detail += "Here are more details about it:\n"
    
    if "director" in fields:
        director = movie_data[fi["director"]]
        if director:
            movie_detail += f"It was directed by {director}.\n"
        else:
            movie_detail += f"Sorry, I couldn't find information about the director of \"{movie_title}\".\n"
    
    if "plot" in fields:
        movie_detail += f"{movie_data[fi["description"]]}"

    if "cast" in fields:
        cast = movie_data[fi["cast"]].lstrip('"').rstrip('"')

        if cast:
            movie_detail += f"The casts are: {cast} \n"
        else:
            movie_detail += f"Sorry, I couldn't find information about the cast of \"{movie_title}\".\n"

    return movie_detail


# %% [markdown]
# #### Processing Queries
# The function `process_query` takes the user input as `query` and uses the `get_movie_index` to retrieve predicted movie by `MultinomialNB`. Then it splits the query in words, makes it full lowercase and lemmatizes it using `WordNetLemmatizer` to reduce possibilities of word form variations. For example, if the user enters 'diREcToRs' instead of 'director', the program will still catch the keyword. Then it checks if the keywords for different fields are available in the lemmatized words list and aopends those to the array. It then gets the asked information and returns it.

# %%

# nltk.download('wordnet')
wnl = WordNetLemmatizer()


DIRECTOR_KEYWORDS = ["director", "directed", "direct"]
PLOT_KEYWORDS = ["plot", "story", "storyline", "describe", "description"]
CAST_KEYWORDS = ["cast", "actor", "act", "acted", "casted"]


def process_query(query):
    
    idx = get_movie_index(query)

    # tokenize the query and lemmatize the words to extract keywords about the movie detail
    words = regexp_tokenize(query, r'\w+')
    lmtzd_words = []
    for word in words:
        lmtzd_words.append(wnl.lemmatize(word.lower()))

    # print(lmtzd_words)                     # uncomment to see the lemmatized words

    # Processing keywords about the movie 
    fields = []
    if check_keywords_in_list(DIRECTOR_KEYWORDS, lmtzd_words):
        fields.append("director")

    if check_keywords_in_list(CAST_KEYWORDS, lmtzd_words):
        fields.append("cast")

    if check_keywords_in_list(PLOT_KEYWORDS, lmtzd_words):
        fields.append("plot")

    # if the user didn't mention anytrhing, tell them about the post
    if not fields:
        fields.append("plot")
        
    return get_movie_detail(idx, fields)




def check_keywords_in_list(keywords, match_list):
    for k in keywords:
        if k in match_list:
            return True
        
    return False



# %% [markdown]
# #### Testing with a sample query

# %%
query = "tell me about the directors and casts of the pianist"
response = process_query(query)
print(response)


# %% [markdown]
# > Dataset source: https://www.kaggle.com/datasets/shivamb/netflix-shows

# %% [markdown]
# # Go!

# %%
print("Bot: Hi! You can ask me about any movie and I will tell you about it. BEEP BOP!", flush=True)
while True:
    query = input()
    print(f"You: {query}" ,flush=True)

    if query == "exit":
        break
    print(f"Bot: {process_query(query)}")


