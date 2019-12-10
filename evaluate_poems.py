from plan import Planner
from rank_words import RankedWords
from poems import Poems
from generateModel import GenerateModel
from paths import data_sxhy_path

import jieba
import json

"""
evaluate generated scores based on similarity with existing poem dataset NOT Rhyme and Tone
"""

ranked_words = RankedWords()
word_scores = dict()
for pair in ranked_words.word_scores:
    word_scores[pair[0]] = pair[1]


def eval_poems(poem, sorted_poem_scores1, sorted_poem_scores2, sorted_poem_scores3):
    rv_score = 0.0

    # process first lien
    target_score = get_score(poem[0])
    target_poem = binarySearch(sorted_poem_scores1, target_score, 1)
    rv_score += abs(target_poem['score2'] - target_score)

    # process 2 lien
    target_score = get_score(poem[1])
    target_poem = binarySearch(sorted_poem_scores2, target_score, 2)
    rv_score += abs(target_poem['score3'] - target_score)

    target_score = get_score(poem[2])
    target_poem = binarySearch(sorted_poem_scores3, target_score, 3)
    rv_score += abs(target_poem['score4'] - target_score)

    return rv_score


def binarySearch(sorted_poem_scores, target_score, num_of_sentence):
    def get_sentence_score(poem_score, num_of_sentence):
        if num_of_sentence == 1:
            return poem_score["score1"]
        elif num_of_sentence == 2:
            return poem_score["score2"]
        elif num_of_sentence == 3:
            return poem_score["score3"]
        else:
            return poem_score["score4"]

    l = 0
    r = len(sorted_poem_scores) - 1

    while (l + 1 < r):
        mid = (l + r) // 2

        if (get_sentence_score(sorted_poem_scores[mid], num_of_sentence) < target_score):
            l = mid
        elif (get_sentence_score(sorted_poem_scores[mid], num_of_sentence) > target_score):
            r = mid
        else:
            return sorted_poem_scores[mid]

    if (abs(get_sentence_score(sorted_poem_scores[l], num_of_sentence) - target_score) > abs(
            get_sentence_score(sorted_poem_scores[r], num_of_sentence) - target_score)):
        return sorted_poem_scores[r]
    else:
        return sorted_poem_scores[l]


class PoemScore:

    def __init__(self, sentences):
        self.first = sentences[0]
        self.second = sentences[1]
        self.third = sentences[2]
        self.fourth = sentences[3]

        self.score1 = get_score(self.first)
        self.score2 = get_score(self.second)
        self.score3 = get_score(self.third)
        self.score4 = get_score(self.fourth)


def get_poem_scores():
    poems = Poems()
    poem_scores = []
    # count=0
    for p in poems:
        # count+=1
        # if count>100:
        #     break
        if len(p) != 4:
            continue
        """
        first.append( (get_score(p[0]), p[0]) )
        second.append( (get_score(p[1]), p[0]) )
        third.append( (get_score(p[2]), p[0]) )
        fourth.append( (get_score(p[3]), p[0]) )
        """
        poem_scores.append(PoemScore(p))

    return poem_scores


def get_score(sentence):
    planner = Planner()
    highest_ranked_word = ''
    highest_ranked_score = 0.0

    for word in jieba.lcut(sentence):
        if word not in word_scores:
            continue
        if word_scores[word] > highest_ranked_score:
            highest_ranked_word = word

    # print("keyword is {w}".format(w=highest_ranked_word))
    keyword = set()
    keyword.add(highest_ranked_word)
    score = 0.0
    count = 0

    for word in planner._expand(keyword):
        if word in word_scores:
            score += word_scores[word]
            count += 1
        # else:
            # print('{w} does not have a score'.format(w=word))

    return score / count


def main():
    planner = Planner()
    generator = GenerateModel(False)
    # print("Start to preprocess score")
    # poem_scores = get_poem_scores()
    # with open("poem_scores.txt", "w") as f:
    #     json_data = json.dumps([score.__dict__ for score in poem_scores])
    #     json.dump(json_data, f)
    # print("score calculation completed")

    with open("poem_scores.txt", 'r') as f:
        poem_scores = json.loads(json.load(f))

    print("process score1...")
    sorted_poem_scores1 = sorted(poem_scores, key=lambda curr_poem: curr_poem['score1'])
    print("process score2...")
    sorted_poem_scores2 = sorted(poem_scores, key=lambda curr_poem: curr_poem['score2'])
    print("process score3...")
    sorted_poem_scores3 = sorted(poem_scores, key=lambda curr_poem: curr_poem['score3'])

    avg_score = 0
    num = 500

    with open("scores_result_100.txt", "w") as f:
        for line in open(data_sxhy_path):
            listWords = line.split()
        for i in range(num):
            keyword = listWords[i]
            keywords = planner.plan(keyword)
            poem = generator.generate(keywords)
            score = eval_poems(poem, sorted_poem_scores1, sorted_poem_scores2, sorted_poem_scores3)
            avg_score += score
            for sentence in poem:
                f.write(sentence)
                f.write("\n")
            f.write("The score of the current poem is:" + str(score))
            f.write("\n")
        f.write("The average score is:" + str(avg_score / num))

if __name__ == '__main__':
    main()
