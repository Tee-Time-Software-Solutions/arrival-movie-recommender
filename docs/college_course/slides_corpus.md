Here is your **concise corpus for Session 1 (Intro + Foundations of Recommender Systems)**, integrating **all core concepts + image insights**.

---

# 📚 **Corpus — Session 1: Recommender Systems Foundations**

## 1. What Recommendation Systems Are

Recommendation systems are **AI systems that predict user preferences** to suggest relevant items. They act like a **personalized salesperson**, matching users with products at the right time.

**Core idea:**

> “Meet unmet, unarticulated user needs” (Satya Nadella mindset)

### From images:

* Logos (Netflix, Amazon, TikTok, etc.) → **Recommenders are ubiquitous across industries** (media, retail, social, gaming).
* Visual emphasis: recommender systems = **core infrastructure, not a feature**.

---

## 2. Business Value & Impact

Recommenders drive **massive revenue and engagement**:

* Amazon → ~35% revenue from recommendations
* Netflix → ~75% content consumption
* YouTube → ~60% clicks
* TikTok → ~90% content discovery
* Spotify → ~10% revenue

👉 Insight:
Recommenders = **highest ROI ML systems** because they directly affect user decisions.

---

## 3. Core Components of a Recommender System

Every system is built on:

* **User** → the consumer
* **Item** → product/content
* **Interaction (feedback)** → signal connecting them

👉 Represented as:

* User–Item interaction matrix (seen visually in CF slide)

---

## 4. Types of Feedback (Data)

### Explicit Feedback

* Ratings, reviews, surveys
* Pros: rich, interpretable
* Cons: sparse, biased, costly

### Implicit Feedback

* Clicks, purchases, watch time
* Pros: abundant, realistic
* Cons: noisy, lacks negative signals

### From images:

* Explicit example: rating UI
* Implicit example: social feed interactions
  👉 Shows **behavioral vs stated preference gap**

---

## 5. Additional Data Sources

### User & Item Features

* User: age, location, demographics
* Item: category, price, attributes
* Helps when interaction data is sparse

### Knowledge-Based Data

* Rules, constraints, knowledge graphs
* Useful for rare/high-cost items (cars, houses)
* Improves explainability

👉 Image insight:

* Knowledge graph (actors, genres, entities) → **semantic relationships enhance recommendations**

---

## 6. Key Challenges

### Cold Start Problem

* **Cold user**: new/no data
* **Cold item**: new product


👉 Critical limitation of collaborative filtering.

---

## 7. Data Transformation (Feature Engineering)

Used to convert raw interactions → model-ready signals:

* **Count** → frequency = affinity
* **Weighted count** → different importance per action
* **Time decay** → recent interactions matter more
* **Weighted time decay** → combine both
* **One-hot encoding** → categorical features
* **Binarization** → convert to 0/1
* **Negative sampling** → generate negative examples

👉 Image insight:

* Tables show transformation from raw logs → **user-item affinity matrix**

---

## 8. Train/Test Splitting Strategies

* **Random split** → simple but unrealistic
* **Stratified split** → avoids cold users/items in test
* **Chronological split** → respects time order

👉 Insight:

* Chronological split = **most realistic for production systems**

---

## 9. Types of Recommendation Algorithms

### Collaborative Filtering (CF)

* Uses **user behavior patterns**
* Example logic:

  * “Users like you also liked X”
* No need for item/user features


👉 Image:

* User-item matrix with missing values → **matrix completion problem**

---

### Content-Based Filtering (CBF)

* Uses **features of users & items**
* Recommends similar items to what user liked


👉 Image:

* Clusters/decision regions → **feature-space similarity**

---

### Key Comparison

| CF                   | CBF                  |
| -------------------- | -------------------- |
| Learns from behavior | Learns from features |
| Suffers cold start   | Handles cold start   |
| No metadata needed   | Needs features       |

---

## 10. Use Cases Across Domains

From tables:

### Retail

* Personalized recs, upsell, cross-sell, chatbot assistant

### Gaming

* Item recommendations, next-best-action

### Media

* Video/music/content recommendation

### Food & Travel

* Location-based recs, assistants

👉 Insight:

* Core task: **“show different items to different users” at scale**

---

## 11. Recommender System Infrastructure

From repo slides:

* **Library** → models + utilities
* **Examples** → end-to-end notebooks
* **Tests** → reliability pipeline

👉 Image insight:

* GitHub repo → real-world production framework
* Emphasis: **engineering + reproducibility matters as much as models**

---

## 12. Key Takeaways (Mental Model)

1. Recommenders = **user–item–interaction problem**
2. Data > algorithm (implicit signals dominate)
3. Two main paradigms:

   * Collaborative (behavior)
   * Content-based (features)
4. Core challenges:

   * Cold start
   * Data sparsity
5. Goal:

   * **Personalization at scale with business impact**

---


Here’s a **concise corpus per session**, keeping all core concepts and also folding in what the diagrams/images are showing.

## Session 06 — SAR, Matrix Factorization, SVD, ALS

**SAR (Smart Adaptive Recommendations / simple item-based memory model)**
SAR is a **memory-based recommender**. It starts from transactional data such as **user, item, time, event**, then builds:

1. a **user-item affinity matrix** capturing how strongly a user likes/interacts with items, and
2. an **item-item similarity matrix** capturing how similar items are.
   Recommendation scores are then computed by combining a user’s affinities with item similarities, and optionally:

* **downweighting old interactions** with a temporal decay,
* **removing already seen items**, and
* returning **top-k recommendations**.
  The page 1 pipeline image shows exactly this flow from raw events to similarity/affinity matrices to ranked recommendations. Page 3’s highlighted example shows the score for one candidate item is a **weighted sum** of the user’s affinities times the candidate’s similarity to each known item.

**Similarity metrics in SAR**
The session highlights several item-item similarity choices:

* **Count**: favors popular items and predictability.
* **Lift**: favors rarer but highly associated items, so better for discoverability/serendipity.
* **Jaccard**: compromise between count and lift.
* **Mutual information**: how much information two item columns share.
* **Lexicographers mutual information**: adjusts MI to reduce low-frequency bias by scaling with co-occurrence.
* **Cosine similarity**: angle-based similarity between item columns.
* **Inclusion index**: overlap-based measure.
  These choices matter because they change whether the recommender behaves more like “popular and safe” or “niche and surprising.”

**Co-occurrence and affinity**
Co-occurrence is presented as a simple way to compute item similarity from the affinity matrix. The slide visualizes how an interaction matrix is transformed into a **co-occurrence matrix** and then normalized into a **similarity matrix**. The affinity matrix itself can encode:

* **explicit feedback** like ratings,
* **implicit feedback** like clicks, views, purchases, each with custom weights,
* and **recency** via a decay factor with half-life interpretation.
  That half-life idea means an event that happened T time units ago gets half the importance of a current event.

**Evaluation and implementation of SAR**
The SAR slides show a MovieLens example with **top-k outputs** and ranking metrics such as **MAP, NDCG, Precision@k, Recall@k**. The implementation images show both a **CPU version** and a **PySpark version**, emphasizing that SAR is practical for both local and distributed settings. The “future implementations” slide mentions possible GPU, Dask, and stored-procedure versions.

**Matrix Factorization (model-based recommender)**
Matrix factorization shifts from memory-based methods to **model-based methods**. The core idea is that users and items can each be represented by **latent vectors** in a lower-dimensional space. A predicted rating is approximated by the **inner product** of:

* a **user factor vector** (p_u),
* and an **item factor vector** (q_i).
  The rank or number of latent dimensions (f) controls how expressive the model is. The page 11 image is important: it visually places movies in a latent space with axes like **serious vs. escapist** and **male-oriented vs. female-oriented**, showing that latent factors may sometimes be interpretable, though often they are abstract.

**Optimization problem, SVD, ALS**
Regularized matrix factorization minimizes reconstruction error plus regularization on the latent vectors. Two learning strategies are emphasized:

* **SGD**: update parameters against the gradient.
* **ALS**: alternate solving for users while fixing items, then items while fixing users.
  For **SVD-style recommender models**, the slides add:
* **global mean** (\mu),
* **user bias** (b_u),
* **item bias** (b_i),
* plus latent interaction (q_i^T p_u).
  So prediction becomes baseline bias terms plus interaction terms. The SVD slide also shows the model error and SGD update rules. ALS is presented as an alternative that avoids some expensive gradient computations and can also support **implicit feedback**.

---

## Session 07 — MF Limitations, FM, FFM

**Matrix factorization limitation**
This session starts with the key weakness of plain MF: it models user-item interaction with a **linear inner product**. That can be too simple to capture richer patterns in user behavior. The highlighted page 1 figure is important: it shows that closeness in the learned latent space may distort true similarity/ranking relationships from the original interaction matrix. So MF is powerful but not always expressive enough.

**Factorization Machines (FM)**
FM extends MF by modeling **second-order feature interactions**. Instead of only user and item IDs, the input is one big **feature vector** that can include:

* user identity,
* item identity,
* item/user/context features,
* side information.
  The model has:
* a global bias,
* first-order weights,
* and pairwise interactions approximated through low-dimensional latent vectors.
  This is the key trick: instead of learning every pairwise cross term directly, FM factorizes them, which makes learning efficient and scalable. The page 2 diagram shows a single vector containing user, item, other movie features, and time/context features, all entering one FM formula.

**Field-aware Factorization Machines (FFM)**
FFM uses the same basic idea as FM, but improves it by making latent vectors **field-aware**. A “field” is a feature group, such as user, item, device, context, genre, etc. Instead of one shared embedding per feature for all interactions, FFM learns **different latent representations depending on which field the other feature belongs to**.
Why this helps: interactions between features from different categories may behave differently, and one shared representation can generalize poorly. FFM is especially common in CTR/prediction tasks with many sparse categorical features.

**Implementation perspective**
The implementation slides list libraries like **libfm, libffm, xlearn, Vowpal Wabbit FM**, and Microsoft recommenders integrations. The xLearn image shows a typical training setup with task type, learning rate, regularization, epochs, optimizer, and separate train/validation/test usage. So the practical message is: FM/FFM are not just theory; they are standard tools for sparse tabular recommendation/ranking tasks.

---

## Session 08 — Trees, Random Forest, GBDT, LightGBM, XGBoost

**Decision Trees**
A decision tree is a model for **classification or regression** that recursively splits feature space into regions. The intuition is to keep splitting until classes are separated well. Trees are:

* intuitive,
* easy to explain,
* good with categorical variables,
* and do **not require feature normalization**.
  But they **overfit easily** if grown too deep. The images on page 5 show how increasing tree depth creates increasingly fine rectangular decision boundaries and more leaves.

**Purity criteria: Gini and Entropy**
Training a tree means choosing splits that maximize node purity or reduce impurity. Two measures are emphasized:

* **Gini index**: variance/impurity style measure, faster to compute.
* **Entropy**: uncertainty/information measure, slower than Gini.
  Both are zero when a node is pure. These are the standard criteria for deciding the best split.

**Random Forest**
Random Forest is an ensemble of decision trees designed to reduce overfitting. It uses **bagging**:

* sample data into many bootstrap datasets,
* train one tree per bag,
* combine predictions across trees.
  Key ideas:
* sampling is **with replacement**,
* training can be done **in parallel**,
* each tree still uses usual split criteria like Gini or entropy.
  The example images show less brittle, more averaged decision boundaries than single trees.

**Gradient Boosting Decision Trees (GBDT)**
GBDT is another tree ensemble, but unlike bagging it builds trees **sequentially**, with each new tree trying to correct residual errors from earlier trees. The course notes mention:

* weak learners combined additively,
* popular frameworks: **XGBoost** and **LightGBM**,
* two tree-growth strategies: **level-wise** and **leaf-wise**,
* two split strategies: **exact split** and **histogram approximation**.
  This makes GBDT one of the strongest families for structured/tabular data.

**LightGBM**
LightGBM is a GBDT implementation optimized for speed and scale. Main points:

* uses **additive training on residuals**,
* uses a **Taylor expansion** of the loss with gradients and Hessians,
* grows trees **leaf-wise** rather than level-wise,
* and uses **histogram-based split approximation**.
  Leaf-wise growth is often faster and stronger on large data, but can overfit more easily without tuning. The page 11 image contrasts level-wise growth with leaf-wise growth visually. LightGBM also supports **distributed training and GPU**.

**Exact vs histogram split**
A core computational issue in GBDT is split finding:

* **Exact split** scans many values and is more accurate but slower.
* **Histogram split** bins values, making training much faster and usually with little accuracy loss on large datasets.
  This distinction matters because it explains why frameworks like LightGBM scale so well.

**XGBoost vs LightGBM vs Spark integration**
The session distinguishes:

* **XGBoost**: supports exact and histogram split, written in C++, multi-language APIs, distributed and GPU support.
* **LightGBM**: histogram-based, C++, multi-language APIs, distributed and GPU support, many hyperparameters.
* **SynapseML LightGBM on Spark**: distributed LightGBM over Spark workers using local histograms that are merged to find global best splits.
  The Spark slides emphasize distributed histogram construction and synchronization of best splits.

---

## Session 09 — BPR, Pointwise vs Pairwise Loss

**Bayesian Personalized Ranking (BPR)**
BPR is a recommender optimization approach focused on **ranking**, especially for **implicit feedback** data such as views, clicks, and purchases. Its main insight is that “not interacted” does **not necessarily mean negative**. Instead of predicting absolute labels, BPR learns that for a given user:

* observed items should rank **above** unobserved items.
  This makes BPR a ranking-first objective rather than a rating/regression objective.

**Why pointwise loss is problematic**
The pointwise approach treats observed interactions as 1 and everything else as 0. The slide’s highlighted warning is the key concept: many of the 0s are not true negatives; they are just **unknown**. So pointwise training can push the model to incorrectly learn that all unseen items are bad. The page 2 image illustrates this by turning a matrix with positives and unknowns into a binary matrix, losing uncertainty.

**Pairwise loss idea**
Pairwise loss reformulates the task into preferences:

* user (u) prefers item (i) over item (j).
  For each user, you generate item pairs. A plus means (i \succ j), a minus means the reverse, and question marks indicate unknown relationships. Advantages noted in the slides:
* training data directly matches the ranking objective,
* unknown-vs-negative ambiguity is handled better,
* train and test are cleaner from a ranking perspective,
* and the method also extends to explicit feedback by converting ratings into preference pairs.
  The page 3 visual makes this especially clear by showing per-user preference matrices.

**BPR training objective**
BPR is framed probabilistically:

* maximize posterior probability of parameters,
* likelihood modeled with a **sigmoid** over pairwise score differences,
* Gaussian prior over parameters,
* yielding an objective with log-sigmoid terms plus regularization.
  So conceptually, BPR trains the model to increase the score gap between preferred and non-preferred items. This pairwise loss can be used with MF-style models or deep recommenders.

---

## Session 10a — Neural Collaborative Filtering (NCF)

**Why NCF exists**
NCF builds on the limitation of matrix factorization: the inner product is too restrictive because it only captures simple linear interactions. NCF uses neural networks to learn **more flexible, nonlinear user-item interaction functions**.

**GMF (Generalized Matrix Factorization)**
GMF is the neural extension of MF. It still uses user and item embeddings, but instead of a fixed inner product, it adds a neural/output layer whose weights are learned from data. If you choose identity activation and all-ones output weights, you recover classic MF. So GMF is best understood as **MF generalized into a trainable nonlinear setting**.

**MLP branch**
The MLP model concatenates user and item embeddings and passes them through multiple hidden layers with nonlinear activations. This gives the model:

* more flexibility,
* deeper interaction modeling,
* and the ability to learn patterns that a simple dot product cannot capture.

**NeuMF / fusion idea**
The page 1 architecture image is the main visual concept: NCF combines:

* a **GMF branch** for element-wise product style interaction,
* an **MLP branch** for nonlinear transformation after concatenation,
* then fuses them into a shared output layer.
  This hybrid is often called **NeuMF**. It allows the model to benefit from both structured factorization-like interaction and richer nonlinear modeling.

**Loss and implementation details**
The session specifies:

* **binary cross-entropy** loss,
* TensorFlow implementation,
* built-in **negative sampling**,
* batch loading with shuffling,
* **leave-one-out evaluation**,
* major hyperparameters: latent factor dimension, layer sizes, epochs.
  So NCF is especially aligned with implicit feedback recommendation where one predicts interaction probability rather than explicit rating directly.

---

## Session 10b — Wide & Deep

**Core idea**
Wide & Deep jointly trains two components:

* **Wide part = memorization**
* **Deep part = generalization**
  The model is **jointly trained**, not an ensemble of separately trained models. The page 1 image visually contrasts wide-only, wide+deep, and deep-only structures, emphasizing that wide captures direct sparse associations while deep uses dense embeddings and hidden layers to generalize.

**Wide component**
The wide side is a **generalized linear model** over raw and crossed features. These **cross-product transformations** memorize frequent co-occurrences, such as “user type A often likes app/movie type B.” It is strong at exploiting patterns already well represented in historical data.

**Deep component**
The deep side is an **MLP with ReLU activations**. It learns dense embeddings and higher-level nonlinear interactions. This supports generalization via shared representation and correlation transitivity, which can improve coverage and diversity by recommending less obvious but related items.

**Why combine them**
Wide-only systems memorize but may fail to generalize to rare/unseen combinations. Deep-only systems generalize but may miss strong memorized sparse crosses. Wide & Deep combines both strengths in one trainable architecture, especially useful in large-scale recommendation systems such as app recommendation.

**Implementation details from the slides**
The implementation section shows:

* **multi-hot encoding** of genres using `MultiLabelBinarizer`,
* sparse item/user features plus item feature arrays,
* embeddings for wide and deep parts,
* a deep tower with **linear layers, ReLU, batch normalization, dropout**,
* joint fitting with separate optimizers/settings for wide and deep parts.
  The training plots on page 6 show error/ranking metrics improving across epochs, reinforcing that the model is optimized end-to-end.

If you want, I can turn this into a **clean exam-study sheet** with only definitions, formulas, and compare/contrast points.
