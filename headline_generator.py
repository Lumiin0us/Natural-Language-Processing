from transformers import T5ForConditionalGeneration,T5Tokenizer
from transformers import AutoModelWithLMHead, AutoTokenizer
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st 
import re 
import torch 

model = T5ForConditionalGeneration.from_pretrained("Michau/t5-base-en-generate-headline")
tokenizer = T5Tokenizer.from_pretrained("Michau/t5-base-en-generate-headline")

mrm_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
mrm_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

jules_tokenizer = AutoTokenizer.from_pretrained("JulesBelveze/t5-small-headline-generator")
jules_model = T5ForConditionalGeneration.from_pretrained("JulesBelveze/t5-small-headline-generator")
# rouge = Rouge()
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# bert_model_name = "bert-base-uncased"
# bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
# bert_model = AutoModel.from_pretrained(bert_model_name)

# def compute_bert_embedding(text):
#     # Tokenize input text
#     input_ids = bert_tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    
#     # Obtain BERT embeddings
#     with torch.no_grad():
#         output = bert_model(input_ids)
    
#     # Extract embeddings from the last layer
#     embeddings = output.last_hidden_state.mean(dim=1)  # Mean pooling across tokens
    
#     return embeddings

# def choose_most_similar_title(description, titles):
#     max_similarity = -1
#     most_similar_title = None
    
#     # Compute BERT embeddings for the description
#     description_embedding = compute_bert_embedding(description)
    
#     # Iterate through titles and compute similarity
#     for title in titles:
#         # Compute BERT embeddings for the title
#         title_embedding = compute_bert_embedding(title)
        
#         # Compute cosine similarity between title and description embeddings
#         similarity_score = torch.cosine_similarity(description_embedding, title_embedding)
        
#         # Update most similar title if necessary
#         if similarity_score > max_similarity:
#             max_similarity = similarity_score
#             most_similar_title = title
            
#     return most_similar_title, max_similarity.item()

def generate_title(article):
    text =  "headline: " + article
    encoding = tokenizer.encode_plus(text, return_tensors = "pt", max_length=2048, truncation=True)
    input_ids = encoding["input_ids"]
    attention_masks = encoding["attention_mask"]
    beam_outputs = model.generate(
            input_ids = input_ids,
            attention_mask = attention_masks,
            max_length = 50,
            num_beams = 3,
            do_sample = False,
            # top_k=10,
            early_stopping = False,
        )
    # titles = []
    # for i in range(3):
    #     beam_outputs = model.generate(
    #         input_ids = input_ids,
    #         attention_mask = attention_masks,
    #         max_length = 50,
    #         num_beams = 3,
    #         do_sample = True,
    #         top_k=10,
    #         early_stopping = False,
    #     )
    #     titles.append(tokenizer.decode(beam_outputs[0]))
    # return titles
    return tokenizer.decode(beam_outputs[0])

def generate_title_2(article):
    input_ids = tokenizer(
    [WHITESPACE_HANDLER(article)],
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=384
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]
    summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
    )
    return summary
# def generate_summary(article):
#   input_ids = mrm_tokenizer.encode(article, return_tensors="pt", add_special_tokens=True)

#   generated_ids = mrm_model.generate(input_ids=input_ids, num_beams=3, max_length=200,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=False, truncation=True)

#   preds = [mrm_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

#   return preds[0]
def generate_summary(article):
    article = article[:1024]
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer(article, max_length=100, min_length=50, do_sample=False)

def main():
    st.title("Text Summarization")
    text = st.text_area("Enter your text here:", "")

    if st.button("Generate Summary"):
        if text.strip() == "":
            st.error("Please enter some text.")
        else:
            title = generate_title(text)
            title_2 = generate_title_2(text)
            # most_similar_title, similarity_score = choose_most_similar_title(text, [title, title_2])
            summary = generate_summary(text)
            summary_text = summary[0]['summary_text']
            # st.subheader("Most Similar Title With Score: " + str(similarity_score))
            # st.write(summary_text.replace('<pad>', '').replace('</s>', ''))
            # reference_tokens = tokenizer.tokenize(summary_text)

            # Calculate ROUGE scores for each title
            # max_score = 0
            # best_title = None
            # for title in titles:
            #     # Tokenize and pad the title
            #     title_tokens = tokenizer.tokenize(title)
            #     padded_title_tokens = pad_title(title_tokens, len(reference_tokens))

            #     # Calculate ROUGE scores
            #     scores = rouge.get_scores(padded_title_tokens, reference_tokens)
            #     f1_score = scores[0]['rouge-1']['f']
            #     if f1_score > max_score:
            #         max_score = f1_score
            #         best_title = title

            st.subheader("Generated Title:")
            st.write(title.replace('<pad>', '').replace('</s>', ''))

            st.subheader("Second Title:")
            st.write(title_2)

            st.subheader("Generated Description:")
            # st.write(summary.replace('<pad>', '').replace('</s>', ''))
            st.write(summary_text)

            # st.subheader("Second Description:")

            # st.write(summary.replace('<pad>', '').replace('</s>', ''))
            # st.write(summary_text)
            # st.subheader("All Titles:")

            # st.write(summary.replace('<pad>', '').replace('</s>', ''))
            # st.write(title)

# def pad_title(tokens, target_length):
#     if len(tokens) >= target_length:
#         return tokens[:target_length]
#     else:
#         padding_length = target_length - len(tokens)
#         padded_tokens = tokens + ['<pad>'] * padding_length
#         return padded_tokens

if __name__ == "__main__":
    main()

# As Donald Trump faces dwindling options to pay off a massive fine imposed as a result of losing a fraud case in New York, financial experts say filing for bankruptcy would provide one clear way out of his financial jam. But Trump is not considering that approach, partially out of concern that it could damage his campaign to recapture the White House from President Biden in November, according to four people close to the former president, who spoke on the condition of anonymity to describe sensitive discussions about Trump’s finances. Even though bankruptcy could alleviate his immediate cash crunch, it also carries risks for a candidate who has marketed himself as a winning businessman — and whose greatest appeal to voters, some advisers say, is his financial success.

# A bankruptcy filing by Trump personally or by one of his companies could delay for months or years the requirement that he pay the judgment of nearly half a billion dollars, which with interest is growing by more than $100,000 a day. A federal judge would be charged with the time-consuming task of determining how and when each of his creditors, including the state, would be paid. In the meantime, Trump could focus on his campaign and not the debt.

# Trump does not have the cash to secure a bond that would delay enforcement of the $464 million judgment while he appeals, his lawyers say. No bonding company will accept real estate — which accounts for most of Trump’s wealth — as collateral. If no bond is posted by Monday, New York Attorney General Letitia James (D) could move to seize his assets, including bank accounts or properties such as Trump’s Manhattan office tower at 40 Wall Street. “He’d rather have Letitia James show up with the sheriff at 40 Wall and make a huge stink about it than say he’s bankrupt,” one of the people close to Trump said. “He thinks about what is going to play politically well for him. Bankruptcy doesn’t play well for him, but having her try to take his properties might.” Yet filing for bankruptcy is a maneuver Trump has used before — six times, when extricating himself from a tumultuous foray into the Atlantic City casino business decades ago. On the campaign trail, Trump in years past explained away those corporate bankruptcies, saying he used a tool many savvy investors have employed — and noting that he never had to file personally.

# Were he to file for bankruptcy now, he probably would not have to “pay anything until after the bankruptcy, and that will take several years because of the complexity,” bankruptcy attorney Avi Moshenberg said. However, Moshenberg said, interest would probably continue to accrue during the bankruptcy.

# A Trump spokesman said the plan is to continue fighting in court. In a filing Monday, Trump’s lawyers asked a panel of appeals court judges to waive the bond requirement. The appeals panel has yet to rule.

# “This is a motion to stay the unjust, unconstitutional, un-American judgment from New York Judge Arthur Engoron in a political Witch Hunt brought by a corrupt Attorney General. A bond of this size would be an abuse of the law, contradict bedrock principals of our Republic, and fundamentally undermine the rule of law in New York,” said the spokesman, Steven Cheung. Trump’s lawyers have expressed some optimism to him privately that appellate judges could decide to shrink the size of the bond he is required to post to avoid asset seizure, one of the other people close to the former president said. Trump has polled advisers, lawyers and others in recent days about what he should do if the court doesn’t come to his aid, and he hasn’t yet come to a decision, that person said.

# Last month, New York Supreme Court Justice Arthur Engoron found that Trump, his two eldest sons and two of his executives submitted fraudulent financial data to lenders and insurance companies to secure better deals. Engoron ordered Trump to pay more than $350 million in penalties, plus interest. His two sons were ordered to pay $4 million each.

# To delay enforcement of the New York judgment while he appeals, Trump and his co-defendants must post that amount in cash or a bond — a guarantee that a third party will pay Trump’s bill if he ends up losing. To secure such a bond, they must put up 120 percent of the judgment — or $557 million — plus pay an $18 million fee to the bond-issuing company, according to an affidavit from Gary Giulietti, an insurance broker and personal friend of Trump’s. Giulietti wrote that obtaining a bond of that size was a “practical impossibility.” He did not respond to a request for comment. A lawyer representing Trump’s co-defendants in the case did not immediately respond to a request for comment. That amount is on top of a $91 million bond Trump posted less than two weeks ago to delay enforcement of a judgment in a defamation lawsuit he lost to writer E. Jean Carroll.

# Both Trump and his advisers told others that they believed until recent days they could get a bond in the New York civil fraud case, according to three of the people in his orbit.

# Relief from a state appeals court would be the least painful way out of Trump’s predicament, the people said.

# Longtime finance attorney Richard Porter, a member of the Republican National Committee who is not involved in Trump’s defense, said experienced judges in New York care about the state’s reputation as a financial center with commercially savvy courts. He said he believes that they will view the half-billion-dollar judgment against Trump skeptically. “Appellate judges are likely to both see and be willing to say that the damage number makes no sense.”

# Trump could also find a bank or extremely wealthy individual willing to come to his aid, either by accepting some of his real estate as collateral and helping with a bond, or by lending him money against his properties. However, Trump has few remaining ties with Wall Street banks; based on his most recent financial disclosure, submitted to a federal ethics office in August, he has only about a half-dozen loans remaining from a few banks.

# “His next best bet is to find a liquid billionaire and do a quick buy-and-sell arrangement with him or her,” Porter said. “Then, if I’m advising that billionaire, I’m saying the upside is you can make some money and make friends with a guy who is likely to be president. The downside is, you will be targeted if he loses,” he said, referring to Trump’s political opponents.

# Though he is loath to sell his properties, Trump could try offloading some hotels or golf courses for cash in coming days. Such transactions generally take weeks or months. The appellate court could also order that he do so but give him more time, according to legal experts.

# “This is his worst nightmare from a personal and financial situation,” said journalist Timothy O’Brien, who wrote a biography of Trump and later served as a political adviser to Mike Bloomberg, the billionaire who ran for president as a Democrat in 2020. With few options available, O’Brien said he expected Trump to lash out even more aggressively in public.

# “He’ll take it to his base,” he said.

# Trump issued a statement Monday night attacking James and the court as tools of the Democratic Party and calling the bond amount “unprecedented, and practically impossible for ANY Company, including one as successful as mine.”

# “The Bonding Companies have never heard of such a bond, of this size, before, nor do they have the ability to post such a bond, even if they wanted to,” he said.

# Some of the people in Trump’s orbit think filing for bankruptcy makes financial sense — even if it could be politically problematic.

# “What is happening to him and his businesses right now is exactly why the bankruptcy code exists,” said one of the people. Filing for bankruptcy could allow him to put a hold on not only the penalties in the fraud case but also any potential liabilities from civil cases surrounding his role in the Jan. 6 attack on the Capitol. “It would be a fresh start,” the person said.

# While Trump isn’t considering bankruptcy now, he has been known among longtime aides to change his mind.

# How much protection filing for bankruptcy would provide Trump — and how long that protection would last — depends in part on whether Trump files personally or on behalf of one of his companies. Experts said personal bankruptcy would almost certainly pause enforcement of the New York judgment for Trump and his co-defendants, including his business entities. But that may be a particularly unappealing option politically.

# He could also choose to file for bankruptcy on behalf of one of his corporate entities.

# Because Trump’s companies are tightly intertwined with his own finances, a bankruptcy judge could rule that Trump is personally protected in the bankruptcy process — and thus not required to pay the penalty immediately — even if just one of his companies files for protection. But even if a bankruptcy judge doesn’t rule that way, the enormous New York judgment would probably be paused while the court took time to reach a decision. To drag the process out as much as possible, Trump could wait to file until the attorney general moves to begin enforcement. It’s not clear whether she would do that immediately if Trump fails to post a bond next week, or whether she would wait until the appeals court rules on his request to waive the bond requirement.

# “If you look at this like a football game, he could let the play clock run down to one second before he calls timeout,” Georgetown law professor Adam J. Levitin said.

# Trump has already lost a measure of corporate independence as a result of the New York civil fraud case. A court-appointed monitor has been overseeing the former president’s businesses since late 2022, and in February, the judge decided that the businesses must get the monitor’s approval before submitting financial information to banks or other third parties.

# Filing for bankruptcy would mean forgoing more control over his business and may force him to make undesirable sales or other transactions down the road.

# He would also have to explain himself on the campaign trail, but he has some experience with that.

# During a 2015 GOP primary debate, moderator Chris Wallace asked Trump why he should be trusted with the nation’s finances given his bankruptcies. Six of his companies filed for bankruptcy in the 1990s and early 2000s after his Atlantic City casinos fell into debt.

# Trump replied that he had started hundreds of companies and only had to use bankruptcy protection a handful of times, much the way other successful business executives had. He pointed out that other Atlantic City gaming companies had filed for protection as well. The banks he owed, he said, were “killers.”

# “I have used the laws of this country just like the greatest people that you read about every day in business have used the laws of this country, the chapter laws, to do a great job for my company, for myself, for my employees, for my family,” he said.