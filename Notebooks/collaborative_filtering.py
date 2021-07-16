# Code for building the book recommendation system and Web App

import pandas as pd
import numpy as np
from math import sqrt
import random
import streamlit as st

ratings = pd.read_csv('../Capstone_Books/Data/ratings_cleaned.csv')
books = pd.read_csv('../Capstone_Books/Data/books_cleaned.csv')

st.set_page_config(initial_sidebar_state="expanded")

# Make Web App
def app():

    # Write in title and first instructions on Web App
    st.title('Read-y Books Recommender')
    st.write("Welcome! To receive book recommendations, fill out the information in the left sidebar. \
              When you are finished rating, the page will refresh and give you your recommendations. \
              You can also select the option below to have a recommendation randomly generated.")
    choice = st.radio(' ', ['Input your own book ratings', 'Or randomly generate a recommendation'])

    # create empty list to store pairs of book title and rating
    d = []

    # if user chooses to input their own data
    if choice == 'Input your own book ratings':

        # let's user choose number of books they'd like to rate
        st.sidebar.write('---')
        st.sidebar.title('Rate books below')
        j = st.sidebar.number_input('How many books would you like to rate?', value=3, min_value=1, max_value=50, step=1)
        st.sidebar.write('---')

        # Initialize keys for each potential selectbox
        keys1, keys2 = ['aa','ab','ac','ad','ae','af','ag','ah','ai','aj'], ['ba','bb','bc','bd','be','bf','bg','bh','bi','bj']

        # first element of dropdown list is a placeholder value '-'
        title_list = ['-'] + books['title'].tolist()

        # for loop that appends book title and rating to list and then appends list to dictionary at every iteration
        for i in range(j):
            b = []

            # user selects book title
            words = 'Select book ' + str(i+1)
            book_choice = st.sidebar.selectbox(words, title_list, key=keys1[i])
            if book_choice == '-':
                pass
            else:
                b.append(book_choice)

            # user selects rating
            rating_choice = st.sidebar.slider('Rating', min_value=0, max_value=5, value=0, key=keys2[i])
            st.sidebar.markdown('---')
            if rating_choice == 0:
                pass
            else:
                b.append(rating_choice)

            # if both title and rating are filled in, append these inputs to d
            if len(b) == 2:
                d.append(b)

        # create dataframe of user inputs to be used in collaborative filtering
        if len(d) == j: # ensure that all boxes are filled in
            user_input = pd.DataFrame(d, columns=['title', 'rating'])
            line = 'You rated ' + str(j) + ' books'
            st.markdown('---')
            st.subheader(line)
            st.write(' ')

        else:
            pass

    # if user chooses to have books randomly chosen for them
    elif choice == 'Or randomly generate a recommendation':

        # number of books we will rate
        j = 5

        # for loop that randomly selects 5 books and gives them a rating of 4 or 5
        for i in range(5):
            b = []
            b.append(random.choice(books['title'].tolist()))
            b.append(random.choice([1,2,3,4,5]))
            d.append(b)

        # create dataframe of user inputs to be used in collaborative filtering
        user_input = pd.DataFrame(d, columns=['title', 'rating'])

        st.markdown('---')
        st.subheader("We rated 5 books")
        st.write(' ')

    # create collaborative filtering class
    class Collaborative_Filtering:

        def __init__(self, books, ratings, user_input):
            self.books = books
            self.ratings = ratings
            self.user_input = user_input
            self.books_cf = None
            self.input = None
            self.books_cf_new = None
            self.genre_list = None
            self.user_subset_group = None
            self.pearson_corr_dict = None
            self.pearson_df = None
            self.top_users_rating = None
            self.recommendation_df = None
            self.recommendation = None

        # make dataframe with only essential columns, convert year column to type int and rename
        def books_cfdf(self):
            self.books_cf = self.books[['id', 'title', 'authors', 'original_publication_year', 'genre1', 'genre2', 'genre3']]
            self.books_cf['original_publication_year'] = self.books_cf['original_publication_year'].astype(int)
            self.books_cf = self.books_cf.rename(columns={'original_publication_year': 'year'})
            self.ratings = self.ratings.rename(columns={'book_id': 'id'})
            return self.books_cf

        # filter the books by title, merge df's and drop year column
        def user_input_filter(self):
            input_id = self.books_cfdf()[self.books_cfdf()['title'].isin(self.user_input['title'].tolist())]
            self.input = pd.merge(input_id, self.user_input)
            self.input = self.input.drop('year', axis=1).reset_index(drop=True)
            return self.input

        # remove input books from books_cf
        def remove_input_books(self):
            input_book_list = self.user_input_filter()['id'].tolist()
            self.books_cf_new = self.books_cfdf()[~self.books_cfdf()['id'].isin(input_book_list)]
            return self.books_cf_new

        # create a list of genres
        def add_genres(self):
            self.genre_list = []
            id_list = self.user_input_filter()['id'].tolist()

            # for loop that appends unique genre tags to genre_list
            for item in id_list:
                temp = self.books.loc[self.books['id'] == item]
                for i in range(24,27):
                    a = temp.iloc[0, i]
                    if a not in self.genre_list:
                        self.genre_list.append(a)
            return self.genre_list

        # filter users that have read books that the input has also read
        def user_subset(self):
            user_subset = self.ratings[self.ratings['id'].isin(self.user_input_filter()['id'].tolist())]
            self.user_subset_group = user_subset.groupby(['user_id'])
            # sort so that users with book most in common with the input will have priority
            self.user_subset_group = sorted(self.user_subset_group, key=lambda x: len(x[1]), reverse=True)
            # limit number of users we look through to 100
            self.user_subset_group = self.user_subset_group[0:100]
            return self.user_subset_group

        # calculate the pearson correlations
        def pearson_correlation(self):
            self.pearson_corr_dict = {}

            # for loop that calculates Pearson correlation and stores values in above dict
            for name, group in self.user_subset():

                group = group.sort_values(by='id')
                self.input = self.user_input_filter().sort_values(by='id')
                num_ratings = len(group)

                # store books from input that share book id with books in each group in df
                temp_df = self.input[self.input['id'].isin(group['id'].tolist())]

                # store both ratings in list for calculations
                rating_list = temp_df['rating'].tolist()
                group_list = group['rating'].tolist()

                # calculate each component of Pearson correlation
                Sxx = sum([i**2 for i in rating_list]) - (sum(rating_list)**2 / float(num_ratings))
                Syy = sum([i**2 for i in group_list]) - (sum(group_list)**2 / float(num_ratings))
                Sxy = sum([i*j for i, j in zip(rating_list, group_list)]) \
                    - (sum(rating_list) * sum(group_list) / num_ratings)

                # calculate Pearson corr if Sxx and Syy not 0, else set = 0
                if Sxx != 0 and Syy != 0:
                    self.pearson_corr_dict[name] = Sxy/sqrt(Sxx*Syy)
                else:
                    self.pearson_corr_dict[name] = 0

            return self.pearson_corr_dict

        # convert dictionary to dataframe
        def to_df(self):
            self.pearson_df = pd.DataFrame.from_dict(self.pearson_correlation(), orient='index')
            self.pearson_df.columns = ['similarity_index']
            self.pearson_df['user_id'] = self.pearson_df.index
            self.pearson_df.index = range(len(self.pearson_df))
            return self.pearson_df

        # get top 50 similar users, merge top_users with ratings, then multiply user similarity by user ratings
        def top_users(self):
            top_users = self.to_df().sort_values(by='similarity_index', ascending=False)[0:50]
            self.top_users_rating = top_users.merge(self.ratings, left_on='user_id', right_on='user_id', how='inner')
            self.top_users_rating['weighted_rating'] = self.top_users_rating['similarity_index'] * self.top_users_rating['rating']
            return self.top_users_rating

        # method that calculates weighted average rec score
        def rec_df(self):
            # apply a sum to the top_users after grouping by user_id
            top_users_sum = self.top_users().groupby('id').sum()[['similarity_index','weighted_rating']]
            top_users_sum.columns = ['sum_similarity_index','sum_weighted_rating']

            # find the weighted average and sort values in order of highest weights descending
            self.recommendation_df = pd.DataFrame()
            self.recommendation_df['weighted average rec score'] = top_users_sum['sum_weighted_rating'] \
                                                                 / top_users_sum['sum_similarity_index']

            # return df of recommendations sorted by weighted average rec score
            self.recommendation_df['id'] = top_users_sum.index
            self.recommendation_df = self.recommendation_df.sort_values(by='weighted average rec score', ascending=False)
            return self.recommendation_df

        # drop id column, return final recommendation
        def recommendations(self):
            fives = self.rec_df().loc[self.rec_df()['weighted average rec score'] == 5]
            if len(fives) > 10:
                self.recommendation = self.remove_input_books().loc[self.remove_input_books()['id'].isin(self.rec_df()['id'].head(len(fives)).tolist())]
                count = 0
                scores = []
                for index, row in self.recommendation.iterrows():
                    if row['genre1'] in self.add_genres():
                        count += 5
                    if row['genre2'] in self.add_genres():
                        count += 3
                    if row['genre3'] in self.add_genres():
                        count += 1
                    scores.append(count)
                    count = 0

                self.recommendation['scores'] = scores
                self.recommendation = self.recommendation.sort_values(by=['scores', 'id'], ascending=[False, True])
                self.recommendation = self.recommendation.drop(['genre1', 'genre2', 'genre3', 'scores'], axis=1)
                #self.recommendation = self.recommendation.reset_index(drop=True)
                self.recommendation = self.recommendation.head(10)

            else:
                self.recommendation = self.remove_input_books().loc[self.remove_input_books()['id'].isin(self.rec_df()['id'].head(10).tolist())]
                self.recommendation = self.recommendation.drop(['genre1', 'genre2', 'genre3'], axis=1)
                #self.recommendation = self.recommendation.reset_index(drop=True)

            return self.recommendation

        # method that returns each url of the user inputted books
        def book_images(self):
            input_df = pd.DataFrame.from_dict(self.user_input)
            input_df = self.books[self.books['id'].isin(self.user_input_filter()['id'])]
            image_list = input_df['image_url'].tolist()
            return image_list

        # method that returns each url of the recommended books
        def rec_images(self):
            input_df = pd.DataFrame.from_dict(self.user_input)
            input_df = self.books[self.books['id'].isin(self.recommendations()['id'])]
            image_list = input_df['image_url'].tolist()
            return image_list

    # Finish Web App

    if len(d) == j:
        # instantiate Collaborative_Filtering class
        cf = Collaborative_Filtering(books, ratings, user_input)

        # display book cover, rating, title and author of input books
        for i in range(j):

            col1, mid, col2, mid, col3 = st.beta_columns([1,2,1,1,20])
            with col1:
                st.image(cf.book_images()[i], width=60)
            with col2:
                st.write(str(cf.user_input_filter().values[i][6]))
            with col3:
                st.write(cf.user_input_filter().values[i][1], '-', cf.user_input_filter().values[i][2])

        st.markdown('---')
        st.subheader('Based on the selected books here are some recommendations:')
        st.write(' ')

        images = cf.rec_images()
        recs = cf.recommendations()

        # display book cover, recommendation ranking, title and author of book recommendations
        for i in range(len(cf.recommendations())):

            col1, mid1, col2, mid2, col3 = st.beta_columns([1,2,1,1,20])
            with col1:
                st.image(images[i], width=60)
            with col2:
                st.write(str(i+1))
            with col3:
                st.write(recs.values[i][1], '-', str(recs.values[i][2]))
