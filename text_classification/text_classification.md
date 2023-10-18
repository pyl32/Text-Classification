```python
import os
import string
import pandas as pd
from statistics import stdev, mean
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout, Activation, Conv1D, MaxPooling1D, LSTM
```

# Text Classification

## Introduction
This project build a classifier to analyze the sentiment of reviews using Keras1 and Python. The text data store in two folders: one folder involves positive reviews, and one folder involves negative reviews.

## Data Exploration and Pre-processing


First, I use binary encoding for the sentiments , i.e y = 1 for positive sentiments and y = −1 for negative sentiments. Since the provided data are pretty clean, we can remove the punctuation and numbers from the data for further analysis.



```python
#pos = os.listdir("../data/pos")
#neg = os.listdir("../data/neg")

from google.colab import drive
drive.mount('/content/drive')
drive_folder = '/content/drive/My Drive/552project/data'

pos = os.listdir(drive_folder + '/pos/')
neg = os.listdir(drive_folder + '/neg/')

data = []
punc_dig = string.punctuation + string.digits

for path in pos:
    f = open("/content/drive/My Drive/552project/data/pos/" + path)
    review = f.readlines()
    f.close()
    for i in range(len(review)):
        review[i] = review[i].translate(str.maketrans('', '', punc_dig))
    data.append([int(path[2:5]), " ".join(review), 1])

for path in neg:
    f = open("/content/drive/My Drive/552project/data/neg/" + path)
    review = f.readlines()
    f.close()
    for i in range(len(review)):
        review[i] = review[i].translate(str.maketrans('', '', punc_dig))
    data.append([int(path[2:5]), " ".join(review), -1])

```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).


The name of each text file starts with cv number. Here, I split the data use text files 0-699 in each class for training and 700-999 for testing.



```python
data = pd.DataFrame(data, columns = ["num", "text", "class"])
```


```python
train = data[data["num"] < 700]
test = data[data["num"] >= 700]
```

Count the number of unique words in the whole dataset (train + test) and print it out.


```python
# resource: https://stackoverflow.com/questions/18936957/count-distinct-words-from-a-pandas-data-frame

results = set()
data['text'].str.lower().str.split().apply(results.update)
print("number of unique words: ", len(results))
```

    number of unique words:  46830



```python
#result = data.text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0)
```

Calculate the average review length and the standard deviation of review lengths. Report the results.



```python
#data0 = data.replace("\n", " ")
```


```python
review_len = []

for i in data["text"]:
    review_len.append(len(i.split()))

print("average: ", mean(review_len))
print("standard deviation: ", stdev(review_len))
```

    average:  644.3555
    standard deviation:  285.0511431249635


### Visualization
Plot the histogram of review lengths.



```python
plt.hist(review_len)
```




    (array([ 75., 537., 743., 417., 136.,  53.,  23.,  13.,   0.,   3.]),
     array([  16. ,  250.7,  485.4,  720.1,  954.8, 1189.5, 1424.2, 1658.9,
            1893.6, 2128.3, 2363. ]),
     <BarContainer object of 10 artists>)




    
![png](output_15_1.png)
    


### NLP/Deep Learning
To represent each text (= data point), there are many ways. In NLP/Deep Learning terminology, this task is called tokenization. It is common to represent text using popularity/ rank of words in text. The most common word in the text will be represented as 1, the second most common word will be represented as 2, etc. Tokenize each text document using this method.



```python
token = Tokenizer()
token.fit_on_texts(data["text"])
pop = token.word_index
pop
```




    {'the': 1,
     'a': 2,
     'and': 3,
     'of': 4,
     'to': 5,
     'is': 6,
     'in': 7,
     'that': 8,
     'it': 9,
     'as': 10,
     'with': 11,
     'for': 12,
     'his': 13,
     'this': 14,
     'film': 15,
     'but': 16,
     'he': 17,
     'i': 18,
     'on': 19,
     'are': 20,
     'by': 21,
     'be': 22,
     'its': 23,
     'an': 24,
     'not': 25,
     'one': 26,
     'movie': 27,
     'who': 28,
     'from': 29,
     'at': 30,
     'was': 31,
     'have': 32,
     'has': 33,
     'her': 34,
     'you': 35,
     'they': 36,
     'all': 37,
     'so': 38,
     'like': 39,
     'about': 40,
     'out': 41,
     'more': 42,
     'when': 43,
     'which': 44,
     'their': 45,
     'up': 46,
     'or': 47,
     'what': 48,
     'some': 49,
     'just': 50,
     'if': 51,
     'there': 52,
     'she': 53,
     'him': 54,
     'into': 55,
     'even': 56,
     'only': 57,
     'than': 58,
     'no': 59,
     'we': 60,
     'good': 61,
     'most': 62,
     'time': 63,
     'can': 64,
     'will': 65,
     'story': 66,
     'films': 67,
     'been': 68,
     'would': 69,
     'much': 70,
     'also': 71,
     'characters': 72,
     'other': 73,
     'get': 74,
     'character': 75,
     'do': 76,
     'them': 77,
     'very': 78,
     'two': 79,
     'first': 80,
     'after': 81,
     'see': 82,
     'well': 83,
     'because': 84,
     'way': 85,
     'make': 86,
     'any': 87,
     'does': 88,
     'really': 89,
     'had': 90,
     'too': 91,
     'while': 92,
     'how': 93,
     'little': 94,
     'life': 95,
     'where': 96,
     'were': 97,
     'plot': 98,
     'off': 99,
     'people': 100,
     'movies': 101,
     'then': 102,
     'me': 103,
     'could': 104,
     'my': 105,
     'bad': 106,
     'scene': 107,
     'never': 108,
     'being': 109,
     'these': 110,
     'over': 111,
     'best': 112,
     'new': 113,
     'doesnt': 114,
     'many': 115,
     'man': 116,
     'scenes': 117,
     'such': 118,
     'dont': 119,
     'know': 120,
     'through': 121,
     'hes': 122,
     'great': 123,
     'another': 124,
     'here': 125,
     'love': 126,
     's': 127,
     'action': 128,
     'go': 129,
     'us': 130,
     'director': 131,
     'something': 132,
     'end': 133,
     'still': 134,
     'back': 135,
     'seems': 136,
     'made': 137,
     'those': 138,
     'work': 139,
     'theres': 140,
     'makes': 141,
     'before': 142,
     'however': 143,
     'now': 144,
     'big': 145,
     'years': 146,
     'few': 147,
     'world': 148,
     'between': 149,
     'every': 150,
     'though': 151,
     'seen': 152,
     'better': 153,
     'enough': 154,
     'around': 155,
     'take': 156,
     'both': 157,
     'performance': 158,
     'why': 159,
     'audience': 160,
     'down': 161,
     'going': 162,
     'isnt': 163,
     'same': 164,
     'should': 165,
     'gets': 166,
     'role': 167,
     'real': 168,
     'may': 169,
     'things': 170,
     'your': 171,
     'think': 172,
     'last': 173,
     'actually': 174,
     'funny': 175,
     'look': 176,
     'own': 177,
     'almost': 178,
     'say': 179,
     'thing': 180,
     'nothing': 181,
     'comedy': 182,
     'fact': 183,
     'although': 184,
     'played': 185,
     'thats': 186,
     'right': 187,
     'find': 188,
     'john': 189,
     'come': 190,
     'since': 191,
     'did': 192,
     'script': 193,
     'cast': 194,
     'plays': 195,
     'long': 196,
     'young': 197,
     'ever': 198,
     'comes': 199,
     'old': 200,
     'actors': 201,
     'original': 202,
     'part': 203,
     'show': 204,
     'without': 205,
     'acting': 206,
     'each': 207,
     'again': 208,
     'star': 209,
     'least': 210,
     'lot': 211,
     'point': 212,
     'takes': 213,
     'quite': 214,
     'himself': 215,
     'during': 216,
     'away': 217,
     'course': 218,
     'goes': 219,
     'cant': 220,
     'minutes': 221,
     'interesting': 222,
     'effects': 223,
     'three': 224,
     'im': 225,
     'year': 226,
     'screen': 227,
     'might': 228,
     'family': 229,
     'guy': 230,
     'rather': 231,
     'anything': 232,
     'day': 233,
     'far': 234,
     'place': 235,
     'must': 236,
     'watch': 237,
     'once': 238,
     'our': 239,
     'yet': 240,
     'didnt': 241,
     'seem': 242,
     'always': 243,
     'fun': 244,
     'times': 245,
     'instead': 246,
     'trying': 247,
     'bit': 248,
     'special': 249,
     'making': 250,
     'give': 251,
     'want': 252,
     'sense': 253,
     'job': 254,
     'picture': 255,
     'kind': 256,
     'having': 257,
     'wife': 258,
     'set': 259,
     'home': 260,
     'probably': 261,
     'series': 262,
     'help': 263,
     'along': 264,
     'becomes': 265,
     'pretty': 266,
     'everything': 267,
     'hollywood': 268,
     'sure': 269,
     'dialogue': 270,
     'men': 271,
     'together': 272,
     'american': 273,
     'woman': 274,
     'actor': 275,
     'become': 276,
     'gives': 277,
     'hard': 278,
     'money': 279,
     'given': 280,
     'high': 281,
     'black': 282,
     'whole': 283,
     'watching': 284,
     'wants': 285,
     'music': 286,
     'got': 287,
     'feel': 288,
     'perhaps': 289,
     'done': 290,
     'especially': 291,
     'death': 292,
     'less': 293,
     'next': 294,
     'moments': 295,
     'sex': 296,
     'everyone': 297,
     'play': 298,
     'looks': 299,
     'completely': 300,
     'city': 301,
     'looking': 302,
     'reason': 303,
     'whose': 304,
     'horror': 305,
     'shows': 306,
     'rest': 307,
     'until': 308,
     'performances': 309,
     'different': 310,
     'simply': 311,
     'james': 312,
     'father': 313,
     'friends': 314,
     'ending': 315,
     'couple': 316,
     'put': 317,
     'case': 318,
     'several': 319,
     'mind': 320,
     'theyre': 321,
     'evil': 322,
     'left': 323,
     'anyone': 324,
     'michael': 325,
     'night': 326,
     'human': 327,
     'shes': 328,
     'small': 329,
     'entire': 330,
     'itself': 331,
     'humor': 332,
     'girl': 333,
     'getting': 334,
     'lost': 335,
     'turns': 336,
     'line': 337,
     'main': 338,
     'found': 339,
     'use': 340,
     'problem': 341,
     'half': 342,
     'begins': 343,
     'true': 344,
     'either': 345,
     'stars': 346,
     'mother': 347,
     'soon': 348,
     'ive': 349,
     'unfortunately': 350,
     'later': 351,
     'final': 352,
     'idea': 353,
     'name': 354,
     'someone': 355,
     'school': 356,
     'comic': 357,
     'town': 358,
     'thought': 359,
     'wrong': 360,
     'else': 361,
     'based': 362,
     'friend': 363,
     'alien': 364,
     'tries': 365,
     'group': 366,
     'second': 367,
     'against': 368,
     'house': 369,
     'written': 370,
     'david': 371,
     'used': 372,
     'sequence': 373,
     'keep': 374,
     'dead': 375,
     'often': 376,
     'certainly': 377,
     'works': 378,
     'relationship': 379,
     'believe': 380,
     'called': 381,
     'named': 382,
     'said': 383,
     'despite': 384,
     'playing': 385,
     'behind': 386,
     'head': 387,
     'turn': 388,
     'finally': 389,
     'under': 390,
     'war': 391,
     'maybe': 392,
     'doing': 393,
     'tell': 394,
     'days': 395,
     'kids': 396,
     'able': 397,
     'finds': 398,
     'seeing': 399,
     'nice': 400,
     'perfect': 401,
     'youre': 402,
     'past': 403,
     'hand': 404,
     'including': 405,
     'book': 406,
     'mr': 407,
     'person': 408,
     'shot': 409,
     'lives': 410,
     'boy': 411,
     'run': 412,
     'camera': 413,
     'supposed': 414,
     'live': 415,
     'lines': 416,
     'tv': 417,
     'moment': 418,
     'side': 419,
     'directed': 420,
     'need': 421,
     'starts': 422,
     'fight': 423,
     'car': 424,
     'entertaining': 425,
     'summer': 426,
     'style': 427,
     'running': 428,
     'game': 429,
     'full': 430,
     'worth': 431,
     'dark': 432,
     'worst': 433,
     'face': 434,
     'start': 435,
     'upon': 436,
     'try': 437,
     'matter': 438,
     'kevin': 439,
     'others': 440,
     'nearly': 441,
     'hour': 442,
     'care': 443,
     'son': 444,
     'opening': 445,
     'throughout': 446,
     'example': 447,
     'exactly': 448,
     'violence': 449,
     'video': 450,
     'early': 451,
     'daughter': 452,
     'major': 453,
     'beautiful': 454,
     'review': 455,
     'problems': 456,
     'sequences': 457,
     'short': 458,
     'wasnt': 459,
     'version': 460,
     'production': 461,
     'title': 462,
     'whos': 463,
     'let': 464,
     'robert': 465,
     'obvious': 466,
     'joe': 467,
     'top': 468,
     'classic': 469,
     'screenplay': 470,
     'already': 471,
     'guys': 472,
     'kill': 473,
     'drama': 474,
     'direction': 475,
     'fine': 476,
     'children': 477,
     'eyes': 478,
     'team': 479,
     'order': 480,
     'themselves': 481,
     'roles': 482,
     'simple': 483,
     'hit': 484,
     'knows': 485,
     'question': 486,
     'act': 487,
     'sort': 488,
     'supporting': 489,
     'earth': 490,
     'truly': 491,
     'white': 492,
     'deep': 493,
     'save': 494,
     'boring': 495,
     'sometimes': 496,
     'jack': 497,
     'known': 498,
     'women': 499,
     'beginning': 500,
     'wont': 501,
     'scream': 502,
     'coming': 503,
     'hell': 504,
     'jokes': 505,
     'killer': 506,
     'four': 507,
     'attempt': 508,
     'arent': 509,
     'strong': 510,
     'space': 511,
     'tom': 512,
     'happens': 513,
     'body': 514,
     'york': 515,
     'room': 516,
     'ends': 517,
     'hope': 518,
     'heart': 519,
     'says': 520,
     'jackie': 521,
     'tells': 522,
     'novel': 523,
     'peter': 524,
     'possible': 525,
     'saw': 526,
     'yes': 527,
     'stupid': 528,
     'quickly': 529,
     'genre': 530,
     'five': 531,
     'lead': 532,
     'extremely': 533,
     'manages': 534,
     'girls': 535,
     'wonder': 536,
     'murder': 537,
     'particularly': 538,
     'lee': 539,
     'romantic': 540,
     'level': 541,
     'stop': 542,
     'ship': 543,
     'future': 544,
     'appears': 545,
     'career': 546,
     'involving': 547,
     'worse': 548,
     'voice': 549,
     'involved': 550,
     'mostly': 551,
     'thriller': 552,
     'sets': 553,
     'eventually': 554,
     'police': 555,
     'sound': 556,
     'hours': 557,
     'falls': 558,
     'taking': 559,
     'emotional': 560,
     'attention': 561,
     'result': 562,
     'material': 563,
     'dr': 564,
     'ones': 565,
     'elements': 566,
     'planet': 567,
     'hero': 568,
     'close': 569,
     'lack': 570,
     'bring': 571,
     'child': 572,
     'meet': 573,
     'whats': 574,
     'piece': 575,
     'note': 576,
     'experience': 577,
     'none': 578,
     'fall': 579,
     'van': 580,
     'brother': 581,
     'dog': 582,
     'leads': 583,
     'fiction': 584,
     'fans': 585,
     'living': 586,
     'wild': 587,
     'alone': 588,
     'de': 589,
     'enjoy': 590,
     'theater': 591,
     'battle': 592,
     'obviously': 593,
     'guess': 594,
     'interest': 595,
     'youll': 596,
     'paul': 597,
     'usually': 598,
     'late': 599,
     'feeling': 600,
     'among': 601,
     'taken': 602,
     'laughs': 603,
     'husband': 604,
     'laugh': 605,
     'parents': 606,
     'george': 607,
     'power': 608,
     'aliens': 609,
     'king': 610,
     'mean': 611,
     'happen': 612,
     'attempts': 613,
     'needs': 614,
     'talent': 615,
     'within': 616,
     'number': 617,
     'chance': 618,
     'across': 619,
     'single': 620,
     'deal': 621,
     'brothers': 622,
     'chris': 623,
     'talk': 624,
     'williams': 625,
     'forced': 626,
     'feels': 627,
     'wonderful': 628,
     'success': 629,
     'easy': 630,
     'features': 631,
     'god': 632,
     'whether': 633,
     'history': 634,
     'expect': 635,
     'killed': 636,
     'words': 637,
     'word': 638,
     'feature': 639,
     'premise': 640,
     'television': 641,
     'leave': 642,
     'mission': 643,
     'impressive': 644,
     'science': 645,
     'poor': 646,
     'except': 647,
     'form': 648,
     'giving': 649,
     'tale': 650,
     'seemed': 651,
     'recent': 652,
     'call': 653,
     'oscar': 654,
     'meets': 655,
     'disney': 656,
     'basically': 657,
     'score': 658,
     'surprise': 659,
     'serious': 660,
     'apparently': 661,
     'told': 662,
     'important': 663,
     'filmmakers': 664,
     'crew': 665,
     'entertainment': 666,
     'released': 667,
     'stuff': 668,
     'somehow': 669,
     'easily': 670,
     'parts': 671,
     'robin': 672,
     'computer': 673,
     'happy': 674,
     'change': 675,
     'brings': 676,
     'art': 677,
     'hilarious': 678,
     'am': 679,
     'whom': 680,
     'ryan': 681,
     'credits': 682,
     'local': 683,
     'events': 684,
     'difficult': 685,
     'remember': 686,
     'went': 687,
     'release': 688,
     'working': 689,
     'ago': 690,
     'crime': 691,
     'sequel': 692,
     'certain': 693,
     'wouldnt': 694,
     'oh': 695,
     'lets': 696,
     'using': 697,
     'id': 698,
     'complete': 699,
     'middle': 700,
     'audiences': 701,
     'cool': 702,
     'william': 703,
     'girlfriend': 704,
     'due': 705,
     'runs': 706,
     'batman': 707,
     'ben': 708,
     'effective': 709,
     'turned': 710,
     'return': 711,
     'viewer': 712,
     'ill': 713,
     'reality': 714,
     'suspense': 715,
     'smith': 716,
     'flick': 717,
     'quality': 718,
     'presence': 719,
     'popular': 720,
     'uses': 721,
     'anyway': 722,
     'dramatic': 723,
     'mystery': 724,
     'personal': 725,
     'begin': 726,
     'surprisingly': 727,
     'youve': 728,
     'figure': 729,
     'die': 730,
     'decides': 731,
     'writing': 732,
     'viewers': 733,
     'somewhat': 734,
     'ways': 735,
     'annoying': 736,
     'absolutely': 737,
     'similar': 738,
     'previous': 739,
     'blood': 740,
     'business': 741,
     'shots': 742,
     'light': 743,
     'came': 744,
     'couldnt': 745,
     'read': 746,
     'strange': 747,
     'gone': 748,
     'excellent': 749,
     'means': 750,
     'former': 751,
     'project': 752,
     'latest': 753,
     'sexual': 754,
     'rich': 755,
     'towards': 756,
     'nor': 757,
     'successful': 758,
     'familiar': 759,
     'visual': 760,
     'amazing': 761,
     'leaves': 762,
     'intelligent': 763,
     'following': 764,
     'beyond': 765,
     'leaving': 766,
     'predictable': 767,
     'romance': 768,
     'wars': 769,
     'present': 770,
     'myself': 771,
     'jim': 772,
     'clear': 773,
     'questions': 774,
     'cut': 775,
     'type': 776,
     'starring': 777,
     'kid': 778,
     'definitely': 779,
     'talking': 780,
     'message': 781,
     'add': 782,
     'powerful': 783,
     'party': 784,
     'herself': 785,
     'brilliant': 786,
     'nature': 787,
     'situation': 788,
     'clever': 789,
     'secret': 790,
     'create': 791,
     'opens': 792,
     'stories': 793,
     'felt': 794,
     'red': 795,
     'giant': 796,
     'office': 797,
     'villain': 798,
     'usual': 799,
     'straight': 800,
     'third': 801,
     'smart': 802,
     'actress': 803,
     'cinema': 804,
     'company': 805,
     'scary': 806,
     'cop': 807,
     'bunch': 808,
     'age': 809,
     'learn': 810,
     'doubt': 811,
     'prison': 812,
     'bill': 813,
     'large': 814,
     'thinking': 815,
     'solid': 816,
     'move': 817,
     'rock': 818,
     'water': 819,
     'bob': 820,
     'follows': 821,
     'saying': 822,
     'million': 823,
     'jones': 824,
     'seriously': 825,
     'writer': 826,
     'effect': 827,
     'potential': 828,
     'america': 829,
     'huge': 830,
     'near': 831,
     'plan': 832,
     'unlike': 833,
     'general': 834,
     'animated': 835,
     'realize': 836,
     'likely': 837,
     'follow': 838,
     'perfectly': 839,
     'motion': 840,
     'understand': 841,
     'decent': 842,
     'martin': 843,
     'took': 844,
     'immediately': 845,
     'mark': 846,
     'moving': 847,
     'subject': 848,
     'married': 849,
     'enjoyable': 850,
     'sam': 851,
     'happened': 852,
     'heard': 853,
     'created': 854,
     'agent': 855,
     'stay': 856,
     'filled': 857,
     'above': 858,
     'th': 859,
     'fails': 860,
     'country': 861,
     'merely': 862,
     'points': 863,
     'sweet': 864,
     'exciting': 865,
     'force': 866,
     'slow': 867,
     'overall': 868,
     'break': 869,
     'wanted': 870,
     'escape': 871,
     'bruce': 872,
     'ultimately': 873,
     'neither': 874,
     'appear': 875,
     'dream': 876,
     'impossible': 877,
     'private': 878,
     'directors': 879,
     'brought': 880,
     'richard': 881,
     'mess': 882,
     'inside': 883,
     'trouble': 884,
     'r': 885,
     'wedding': 886,
     'favorite': 887,
     'tim': 888,
     'murphy': 889,
     'liked': 890,
     'fan': 891,
     'otherwise': 892,
     'musical': 893,
     'various': 894,
     'scott': 895,
     'trek': 896,
     'particular': 897,
     'pay': 898,
     'political': 899,
     'keeps': 900,
     'dumb': 901,
     'ten': 902,
     'situations': 903,
     'steve': 904,
     'chase': 905,
     'talented': 906,
     'minute': 907,
     'harry': 908,
     'members': 909,
     'spend': 910,
     'element': 911,
     'truth': 912,
     'society': 913,
     'studio': 914,
     'bond': 915,
     'effort': 916,
     'focus': 917,
     'silly': 918,
     'slightly': 919,
     'earlier': 920,
     'rating': 921,
     'biggest': 922,
     'open': 923,
     'drug': 924,
     'offers': 925,
     'showing': 926,
     'havent': 927,
     'purpose': 928,
     'cannot': 929,
     'park': 930,
     'memorable': 931,
     'soundtrack': 932,
     'eye': 933,
     'fast': 934,
     'frank': 935,
     'totally': 936,
     'mars': 937,
     'cold': 938,
     'english': 939,
     'view': 940,
     'ideas': 941,
     'gun': 942,
     'state': 943,
     'subplot': 944,
     'aspect': 945,
     'wait': 946,
     'ask': 947,
     'government': 948,
     'credit': 949,
     'box': 950,
     'eddie': 951,
     'waste': 952,
     'constantly': 953,
     'actual': 954,
     'entirely': 955,
     'hands': 956,
     'l': 957,
     'law': 958,
     'fear': 959,
     'british': 960,
     'moves': 961,
     'terrible': 962,
     'e': 963,
     'gave': 964,
     'west': 965,
     'convincing': 966,
     'ability': 967,
     'u': 968,
     'thinks': 969,
     'spent': 970,
     'ridiculous': 971,
     'female': 972,
     'typical': 973,
     'cinematography': 974,
     'atmosphere': 975,
     'setting': 976,
     'lots': 977,
     'animation': 978,
     'carter': 979,
     'air': 980,
     'fairly': 981,
     'control': 982,
     'background': 983,
     'suddenly': 984,
     'killing': 985,
     'expected': 986,
     'depth': 987,
     'tension': 988,
     'sees': 989,
     'sit': 990,
     'greatest': 991,
     'critics': 992,
     'army': 993,
     'humans': 994,
     'complex': 995,
     'beauty': 996,
     'brief': 997,
     'violent': 998,
     'amusing': 999,
     'dull': 1000,
     ...}




```python
#data["split_text"] = data['text'].str.lower().str.split()

```


```python
data["token_text"] = token.texts_to_sequences(data["text"])
data["token_text"]
```




    0       [2499, 181, 1, 4195, 6, 289, 26, 4, 1, 62, 372...
    1       [98, 2275, 8122, 6, 2, 1302, 2011, 17, 6, 71, ...
    2       [18, 174, 679, 2, 891, 4, 1, 202, 47, 38, 1497...
    3       [2, 27, 186, 68, 10, 1048, 1988, 46, 10, 1, 10...
    4       [61, 65, 1967, 6, 79, 101, 7, 26, 24, 2859, 15...
                                  ...                        
    1995    [1974, 43, 2, 14919, 46799, 7, 1, 4917, 2243, ...
    1996    [23, 144, 1, 6233, 4, 1, 23074, 4, 1293, 312, ...
    1997    [25627, 11, 1, 9732, 3671, 4, 101, 8, 621, 11,...
    1998    [3, 144, 1, 19667, 1448, 1489, 427, 4, 1225, 3...
    1999    [3082, 196, 495, 3, 50, 1634, 528, 3082, 490, ...
    Name: token_text, Length: 2000, dtype: object



Select a review length L that 70% of the reviews have a length below it.



```python
L = int(np.percentile(review_len, 70))
L
```




    737



Then, truncate reviews longer than L words and zero-pad reviews shorter than L so that all texts (= data points) are of length L.3



```python
data["pad_text"] = list(pad_sequences(data["token_text"], maxlen = L))
data["pad_text"]
```




    0       [109, 1141, 11933, 1194, 4196, 2, 3737, 3641, ...
    1       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    2       [285, 5, 129, 5, 5, 276, 42, 9269, 36, 17378, ...
    3       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    4       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
                                  ...                        
    1995    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    1996    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    1997    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    1998    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    1999    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
    Name: pad_text, Length: 2000, dtype: object



## Word Embeddings


### i. Use tokenized text as inputs to a deep neural network.
However, a recent breakthrough in NLP suggests that more sophisticated representations of text yield better results. These sophisticated representations are called word embeddings. \Word embedding is a term used for representation of words for text analysis, typically in the form of a real-valued vector that encodes the meaning of the word such that the words that are closer in the vector space are expected to be similar in meaning."4. Most deep learning modules (including Keras) provide a convenient way to convert positive integer representations of words into a word embedding by an \Embedding layer." The layer accepts arguments that define the mapping of words into embeddings,including the maximum number of expected words also called the vocabulary size (e.g. the largest integer value). The layer also allows you to specify the dimension for each word vector, called the \output dimension." We would like to use a word embedding layer for this project. Assume that we are interested in the top 5,000 words. This means that in each integer sequence that represents each document, we set to zero those integers that represent words
that are not among the top 5,000 words in the document.5 If you feel more adventurous, use all the words that appear in this corpus. Choose the length of the embedding vector for each word to be 32. Hence, each document is represented as a 32 × L matrix.

### ii. Flatten the matrix of each document to a vector.


```python
input_text = np.array(list(data["pad_text"]))
input_text[input_text >= 5000] = 0
```


```python
#resource: https://www.cnblogs.com/Renyi-Fan/p/13809918.html
model = Sequential()
model.add(Embedding(5000, 32, input_length = L))
model.add(Flatten())
model.compile('rmsprop', 'mse')
print(model.summary)

embed_text = model.predict(input_text)
print(model.layers[0].get_weights())
print(embed_text)
```

    <bound method Model.summary of <tensorflow.python.keras.engine.sequential.Sequential object at 0x160e69df0>>
    [array([[-0.03551153, -0.0312135 ,  0.02762629, ..., -0.00306189,
             0.03902471, -0.01229275],
           [ 0.04268488, -0.04910192, -0.04501715, ...,  0.00510366,
             0.04548797, -0.00195887],
           [ 0.03849134,  0.03062275,  0.03373922, ...,  0.01225618,
            -0.02856541,  0.0476712 ],
           ...,
           [ 0.02141747,  0.04477562, -0.00798661, ..., -0.0272519 ,
             0.02140928,  0.04162479],
           [ 0.03989843,  0.04823219, -0.00834541, ..., -0.02606398,
            -0.00012935, -0.02234277],
           [-0.01735542,  0.02892314,  0.02645124, ..., -0.00246954,
            -0.02422092,  0.02956215]], dtype=float32)]
    [[ 0.03333518 -0.02617295 -0.02261096 ... -0.04879908 -0.02016357
      -0.00410762]
     [-0.03551153 -0.0312135   0.02762629 ... -0.00327522 -0.00395964
       0.02752754]
     [-0.0490008  -0.04394411 -0.0189552  ... -0.00306189  0.03902471
      -0.01229275]
     ...
     [-0.03551153 -0.0312135   0.02762629 ...  0.0123669   0.03612501
      -0.03114016]
     [-0.03551153 -0.0312135   0.02762629 ... -0.01558664  0.01954446
      -0.01011112]
     [-0.03551153 -0.0312135   0.02762629 ... -0.04458426  0.02998788
      -0.02043418]]


## Multi-Layer Perceptron


```python
Data = data[["num", "class"]]
Data["emb_text"] = list(embed_text)
Data["class"] = Data["class"].replace(-1, 0)
```

    <ipython-input-18-5e7af8508837>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Data["emb_text"] = list(embed_text)
    <ipython-input-18-5e7af8508837>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Data["class"] = Data["class"].replace(-1, 0)



```python
Train = Data[Data["num"] < 700]
Test = Data[Data["num"] >= 700]
X_tr = np.array(list(Train["emb_text"]))
X_t = np.array(list(Test["emb_text"]))
Y_tr = Train["class"]
Y_t = Test["class"]
```

### i. Train a MLP with three (dense) hidden layers each of which has 50 ReLUs and one output layer with a single sigmoid neuron. Use a dropout rate of 20% for the first layer and 50% for the other layers. Use ADAM optimizer and binary cross entropy loss (which is equivalent to having a softmax in the output). To avoid overfitting, just set the number of epochs as 2. Use a batch size of 10.


```python
# resource: https://www.jianshu.com/p/d121ae396130?utm_campaign=maleskine&utm_content=note&utm_medium=seo_notes&utm_source=recommendation

model = Sequential()

model.add(Dense(50, activation = 'relu', input_dim = 23584))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
model.fit(X_tr, Y_tr, epochs = 2, batch_size = 10)
```

    Epoch 1/2
    140/140 [==============================] - 2s 9ms/step - loss: 0.7053 - accuracy: 0.4943
    Epoch 2/2
    140/140 [==============================] - 1s 9ms/step - loss: 0.6841 - accuracy: 0.5241





    <tensorflow.python.keras.callbacks.History at 0x1618ebd00>



### ii. Report the train and test accuracies of this model


```python
test_score = model.evaluate(X_t, Y_t, batch_size = 10)
print("test_score: ", mean(test_score))
train_score = model.evaluate(X_tr, Y_tr, batch_size = 10)
print("test_score: ", mean(train_score))
```

    60/60 [==============================] - 0s 1ms/step - loss: 0.6883 - accuracy: 0.5533
    test_score:  0.6208227872848511
    140/140 [==============================] - 0s 1ms/step - loss: 0.6269 - accuracy: 0.6829
    test_score:  0.6548685431480408


## One-Dimensional Convolutional Neural Network:
Although CNNs are mainly used for image data, they can also be applied to text data, as text also has adjacency information. Keras supports one-dimensional convolutions and pooling by the Conv1D and MaxPooling1D classes respectively.

### i. After the embedding layer, insert a Conv1D layer. This convolutional layer has 32 feature maps , and each of the 32 kernels has size 3, i.e. reads embedded word representations 3 vector elements of the word embedding at a time. The convolutional layer is followed by a 1D max pooling layer with a length and stride of 2 that halves the size of the feature maps from the convolutional layer. The rest of the network is the same as the neural network above.


```python
model = Sequential()
model.add(Embedding(5000, 32, input_length = L))
model.add(Conv1D(32,3))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Flatten())
model.compile('rmsprop', 'mse')
print(model.summary)

embed_text = model.predict(input_text)
print(model.layers[0].get_weights())
print(embed_text)
```

    <bound method Model.summary of <tensorflow.python.keras.engine.sequential.Sequential object at 0x17c918a00>>
    [array([[-0.03625785, -0.00394065, -0.01474299, ...,  0.03426755,
             0.03313489, -0.03360778],
           [ 0.0382765 ,  0.00851555, -0.02150942, ...,  0.0178039 ,
             0.01025463, -0.01649923],
           [ 0.00754521,  0.04487438,  0.00597467, ..., -0.03997531,
            -0.01019354,  0.04177039],
           ...,
           [-0.048969  ,  0.03454694, -0.00401114, ...,  0.02102664,
             0.00111048,  0.02649759],
           [-0.03555398, -0.04463471, -0.0326045 , ..., -0.02369057,
            -0.0470906 , -0.02861959],
           [-0.03839446,  0.02004881,  0.02382291, ..., -0.01642703,
            -0.0390697 , -0.02226261]], dtype=float32)]
    [[ 1.16310893e-02  1.59416627e-02  4.05392908e-02 ...  2.37603653e-02
       2.84849163e-02 -9.02268756e-03]
     [ 1.15169855e-02  4.81385097e-04  8.76351260e-03 ...  1.72630902e-02
       2.32951529e-02  9.26091988e-03]
     [-3.18915071e-03  5.93612343e-02  2.09271610e-02 ...  4.64434028e-02
      -8.77431966e-03  4.62909080e-02]
     ...
     [ 1.15169855e-02  4.81385097e-04  8.76351260e-03 ... -3.15948762e-03
      -2.63488218e-02  2.62491424e-02]
     [ 1.15169855e-02  4.81385097e-04  8.76351260e-03 ...  3.67776155e-02
       2.89064739e-02 -2.69080908e-03]
     [ 1.15169855e-02  4.81385097e-04  8.76351260e-03 ...  3.42333689e-03
       4.80349222e-03 -6.62009697e-05]]



```python
Data1 = data[["num", "class"]]
Data1["emb_text"] = list(embed_text)
Data1["class"] = Data1["class"].replace(-1, 0)
```

    <ipython-input-23-2e2d5b954f82>:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Data1["emb_text"] = list(embed_text)
    <ipython-input-23-2e2d5b954f82>:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      Data1["class"] = Data1["class"].replace(-1, 0)



```python
Train = Data1[Data1["num"] < 700]
Test = Data1[Data1["num"] >= 700]
X_tr = np.array(list(Train["emb_text"]))
X_t = np.array(list(Test["emb_text"]))
Y_tr = Train["class"]
Y_t = Test["class"]
```


```python
model = Sequential()

model.add(Dense(50, activation = 'relu', input_dim = 11744))
model.add(Dropout(0.2))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
model.fit(X_tr, Y_tr, epochs = 2, batch_size = 10)
```

    Epoch 1/2
    140/140 [==============================] - 1s 6ms/step - loss: 0.7036 - accuracy: 0.5212
    Epoch 2/2
    140/140 [==============================] - 1s 6ms/step - loss: 0.6949 - accuracy: 0.5070





    <tensorflow.python.keras.callbacks.History at 0x17c647340>



### ii. Report the train and test accuracies of this model.


```python
test_score = model.evaluate(X_t, Y_t, batch_size = 10)
print("test_score: ", mean(test_score))
train_score = model.evaluate(X_tr, Y_tr, batch_size = 10)
print("test_score: ", mean(train_score))
```

    60/60 [==============================] - 0s 898us/step - loss: 0.6886 - accuracy: 0.5567
    test_score:  0.6226412951946259
    140/140 [==============================] - 0s 987us/step - loss: 0.6824 - accuracy: 0.5714
    test_score:  0.6268921792507172


## Long Short-Term Memory Recurrent Neural Network
The structure of the LSTM we are going to use is shown in the following figure.

### i. Each word is represented to LSTM as a vector of 32 elements and the LSTM is followed by a dense layer of 256 ReLUs. Use a dropout rate of 0.2 for both LSTM and the dense layer. Train the model using 10-50 epochs and batch size of 10.


```python
model = Sequential()
model.add(Embedding(5000, 32, input_length = L))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
#model.fit(X_tr, Y_tr, epochs = 10, batch_size = 10)
```




```python
test_score = model.evaluate(X_t, Y_t, batch_size = 10)
print("test_score: ", mean(test_score))
train_score = model.evaluate(X_tr, Y_tr, batch_size = 10)
print("test_score: ", mean(train_score))
```

    WARNING:tensorflow:Model was constructed with shape (None, 737) for input KerasTensor(type_spec=TensorSpec(shape=(None, 737), dtype=tf.float32, name='embedding_4_input'), name='embedding_4_input', description="created by layer 'embedding_4_input'"), but it was called on an input with incompatible shape (10, 11744).
    WARNING:tensorflow:Model was constructed with shape (None, 737) for input KerasTensor(type_spec=TensorSpec(shape=(None, 737), dtype=tf.float32, name='embedding_4_input'), name='embedding_4_input', description="created by layer 'embedding_4_input'"), but it was called on an input with incompatible shape (10, 11744).
    60/60 [==============================] - 21s 333ms/step - loss: 0.6956 - accuracy: 0.1632
    test_score:  0.5965770184993744
    140/140 [==============================] - 48s 344ms/step - loss: 0.6932 - accuracy: 0.5000
    test_score:  0.5965765416622162

