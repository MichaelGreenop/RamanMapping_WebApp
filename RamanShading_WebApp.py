import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans

# Tasks - General functions section



def upload():
    dataframe = []
    uploaded_file = st.file_uploader('files', accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="hidden")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        dataframe.append(df)
      


def home_page():
    
    # Tasks     

    st.markdown("Home page")
    st.sidebar.markdown("Home page")

    st.write("""
    The aim of this app is to provide a method of objectively\n
    choosing shading boundaries within vibrational spectroscopy\n
    images using k-means clustering
    \n
    Index:\n
        Page 1: False-colour shading (single feature example)\n
        Page 2: False-colour shading (multi-feature example)\n
        Page 3: K-means cluster elbow plot\n
        Page 4: K-means directed shading (part 1)\n
        Page 5: K-means directed shading (part 2)\n
        Page 6: K-means directed shading (part 3)\n
        Page 5: PCA hyperspectral shading comparison\n\n

    Common term definitions:\n
        - Single feature)\n
        - Multi-feature)\n
        - False-colour shading\n
        - Hyperspectral)\n
        - Vibrational spectroscopy mapping)
    """)


def page1():
    st.markdown("Page 1: Single feature mapping")
    st.sidebar.markdown("Page 1: Single feature mapping")
    
    st.write('''This page allows the intensity of a single wavenumber, area under a peak, or\n
     ratio of two wavenumbers / peak areas to be visualised, refered to as univariate shading\n
      as only a single feature is mapped over the area.''')

    Start = ''
    End = ''

    Start = st.text_input("Range start: ", key=0)
    End = st.text_input("Range end: ", key=1)

    #dataframe1 = [] 

    if len(Start) > 0:
        if len(End) > 0:
            uploaded_file1 = st.file_uploader('files', accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="hidden")
            if uploaded_file1 is not None:
                data1 = pd.read_csv(uploaded_file1, header=None)
                data1 = np.array(data1)
                if len(data1) > 0:
                    df1 =data1[:,int(Start):int(End)]
                    df1 = pd.DataFrame(df1).T.mean()


                    dim_x = ''
                    dim_y = ''
                    colour = ''

                    dim_x = st.text_input("X dimension: ", key=2)
                    dim_y = st.text_input("Y dimension: ", key=3)
                    
                    
                    if len(dim_x) > 0:
                        if len(dim_y) > 0:
                            dim_x = int(dim_x)
                            dim_y = int(dim_y)
                            img = np.array(df1).reshape((dim_x, dim_y))
                            colour = st.text_input("Colour: ", key=4)
                            if len(colour) > 0:
                                fig1, ax1 = plt.subplots()
                                ax1.imshow(img, colour)
                                ax1.grid(False)    

                                st.write(fig1)
# Tasks:
# 1) You need methods to produce to process and plot the different maps
    

def page2():


    # Tasks:
    
    st.markdown("Page 2: Multi-feature mapping)")
    st.sidebar.markdown("Page 2: Multi-feature mapping")

    st.write("""
    This page will...
    """)




def page3():

    st.markdown("Page 3: K-means elbow plot)")
    st.sidebar.markdown("Page 3: K-means elbow plot")

    st.write("""
    The plot this page produces helps determine the optimal number of clusters.\n
    The "elbow" is K number of clusters where the difference between the points reduces and the line flattens.\n
    CHoosing K from the elbow is a method of deciding on the number of clusters to use. 
    """)

    # Tasks:
    # 1) Use the method defined previously to import the data
    # 2) PLot the figure
    # 3) Add a feature to print out the clusters upto a threshold (percentage change)
    # 4) Add a button to plot those clusters on the elbow plot, to visualise them
    # 5) You will need to add buttons to save images, potentially you can use a repeatable method

    #dataframe3 = []      
        
    uploaded_file3 = st.file_uploader('files', accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="hidden")
    if uploaded_file3 is not None:
        data3 = pd.read_csv(uploaded_file3, header=None)
        data3 = np.array(data3)


        # Error in the x and y of the plot not matching 
    
        if len(data3) > 0:
            st.write(data3.shape)
            
            K = range(1,10)
            distortions = []
            
            for k in K:
                kmeanModel = KMeans(n_clusters=k)
                kmeanModel.fit(data3)
                distortions.append(kmeanModel.inertia_)
     
            fig3, ax3 = plt.subplots()
            ax3.plot(K, distortions, 'bx-')
            ax3.set_xlabel('k')
            ax3.set_ylabel('Distortion')
            ax3.set_title('The Elbow Method showing the optimal k', fontsize = 10)
            ax3.grid(True)    

            st.write(fig3)
                

    
def page4():

    # Tasks:

    st.markdown("Page 4: K-means directed shading (Step 1)")
    st.sidebar.markdown("Page 4: K-means directed shading (Step 1)")

    st.write("""
    The aim of this page is to plot the map using k-means clustering.\n
    
    The averaged spectra for the different clusters can then be compared,
    providing spectral justificaion for cluster colour allocations in step 3 (next page). 
    """)
    Clusters_K = ''
    dim_x = ''
    dim_y = ''
    
    Clusters_K = st.text_input("Number of clusters: ", key=5)
    dim_x = st.text_input("X dimension: ", key=6)
    dim_y = st.text_input("Y dimension: ", key=7)
    

    

    if len(Clusters_K) > 0:
        if len(dim_x) > 0:
            if len(dim_y) > 0:
                dataframe4 = [] 
                uploaded_file4 = st.file_uploader('files', accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="hidden")
                if uploaded_file4 is not None:
                    data4 = pd.read_csv(uploaded_file4, header=None)
                    data4 = np.array(data4)

                    kmeans = KMeans(n_clusters=int(Clusters_K), random_state=0).fit(data4)
                    df = np.array(kmeans.labels_)
                    img = np.array(df).reshape((int(dim_x),int(dim_y)))
            
                    fig4_1, ax4_1 = plt.subplots()
                    img4_1 = ax4_1.imshow(img)
                    ax4_1.grid(False)
                    fig4_1.colorbar(img4_1, ax=ax4_1)
                    
                    st.write(fig4_1)


                    Clusters_0 = ''            
                    Clusters_0 = st.text_input("First averge spectrum cluster Number: ", key=8)
                

                    if len(Clusters_0) > 0:

                        # need to np concat df and data 4
                        df2 = df.reshape((len(df),1))
                        
                        DF = pd.DataFrame(np.concatenate((df2, data4), axis=1)).set_index([0])

                        fig4_2, ax4_2 = plt.subplots()
                        ax4_2.plot(DF.loc[int(Clusters_0)].mean(),label='Cluster No. ' + Clusters_0)
                        
                        
                        if int(Clusters_K) >= int(2):
                            Clusters_1 = ''
                            Clusters_1 = st.text_input("Second averge spectrum cluster Number: ", key=9)
                            if len(Clusters_1) > 0:
                                ax4_2.plot(DF.loc[int(Clusters_1)].mean(),label='Cluster No. ' + Clusters_1)
                                ax4_2.grid(False)
                                ax4_2.legend()
                                st.write(fig4_2)



                        else:
                            ax4_2.grid(False)
                            ax4_2.legend()

                            st.write(fig4_2)

                        
def page5():

    st.markdown("Page 5: K-means directed shading (Step 2)")
    st.sidebar.markdown("Page 5: K-means directed shading (Step 2)")

    st.write("""
    The aim of this page is to take the information from step 1, where the average spectrum for each cluster was inspected, allowing
    molecules (colours) to be assigned to clusters or groups of clusters.\n

    Once colours are assigned, maps (layers) are produced for each molecule,
    ready for combination in step 3 (page 6). Colourbars for the different layers will also be produced at this stage.
    """)

    Clusters_K = ''
    dim_x = ''
    dim_y = ''
    colour = ''
    Cluster = ''
    rng_s = ''
    rng_e = ''
    
    Clusters_K = st.text_input("Number of clusters: ", key=10)
    dim_x = st.text_input("X dimension: ", key=11)
    dim_y = st.text_input("Y dimension: ", key=12)
    
    

    

    if len(Clusters_K) > 0:
        if len(dim_x) > 0:
            if len(dim_y) > 0:
                #dataframe5 = [] 
                uploaded_file5 = st.file_uploader('files', accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="hidden")
                if uploaded_file5 is not None:
                    
                    
                    data5 = pd.read_csv(uploaded_file5, header=None)
                    data5 = np.array(data5)

                    kmeans = KMeans(n_clusters=int(Clusters_K), random_state=0).fit(data5)
                    df = np.array(kmeans.labels_)
                    df2 = df.reshape((len(df),1))  
                    DF = pd.DataFrame(np.concatenate((df2, data5), axis=1)).set_index([0])
                    
                    Cluster = st.text_input("Which clusters do you want to remove?\n Enter as a list (e.g. [x,y,z]): ", key=14)

                    if len(Cluster) > 0:

                        def zero(df, clusters):
                            inds = np.array(df.index)
                            clusters = np.array(clusters)
    
                            for cluster in clusters:
                                for ind in inds:
                                    if ind == cluster:
                                        df.loc[cluster] = 0
                                    else:
                                        st.write('Did not work')
                                        break
    
                            return df
                        

                        collect_numbers = lambda x : [int(i) for i in re.split("[^0-9]", x) if i != ""]
                        Clusters = np.array(collect_numbers(Cluster))
                        
                        def dbl(x):
                            for i in x:
                                st.write(i*2)

  
                        if len(Clusters) > 0:
                            
                            st.write(dbl(Clusters))

                        #    img = zero(DF, Clusters).mean()


                        #    colour = st.text_input("Colour: ", key=13)
                        #    rng_s = st.text_input("Start of spectral range: ", key=15)
                        #    rng_e = st.text_input("End of spectral range: ", key=16)
                        #    if len(colour) > 0:
                        #        if len(rng_s) > 0:
                        #            if len(rng_e) > 0:
                        #                Img = np.array(img).reshape((int(dim_x),int(dim_y)))
                        #            
                        #                fig5_1, ax5_1 = plt.subplots()
                        #                img5_1 = ax5_1.imshow(Img, cmap=colour)
                        #                ax5_1.grid(False)
                        #                fig5_1.colorbar(img5_1, ax=ax5_1)
                        #            
                        #                st.write(fig5_1)
                    

def page6():

    # Tasks:

    st.markdown("Page 6: K-means directed shading (Step 3)")
    st.sidebar.markdown("Page 6: K-means directed shading (Step 3)")

    st.write("""
    The aim of this page is to combine the layers produced in step 2.\n

    The layer backgrounds will first be made transparent, and then combined
    to form the final image.\n\n 
    """)



def page7():

    # Task 1: Use the method defined previously to import the data
    # Task 2: Add in a CEV curve, and the capacity to output the number of PC's above a threshold
    # Task 3: PCA to reduce the dimensions and plot the hyperspectral image
    # Task 4: Add a the capacity to look at the loading plot
    
    st.markdown("Page 7: PCA hyperspectral imaging")
    st.sidebar.markdown("Page 7: PCA hyperspectral imaging")

    st.write('This page allows for the production of\n PCA hyperspecral images from Raman maps')

    PC = ''
    
    PC = st.text_input("Principal component: ", key=17)
    
    dataframe2 = [] 

    if len(PC) > 0:
        uploaded_file5 = st.file_uploader('files', accept_multiple_files=False, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="hidden")
        if uploaded_file5 is not None:
            data5 = pd.read_csv(uploaded_file5, header=None)
            data5 = np.array(data5)
            if len(data5) > 0:
                pca = PCA().fit(data5)
                
                pca2 = PCA(n_components=16)
                principalComponents = pca2.fit_transform(data5)
                principalDf = np.array(pd.DataFrame(data = principalComponents))

                dim_x = st.text_input("X Dimension: ", key=18)
                dim_y = st.text_input("Y Dimension: ", key=19)

                if len(dim_x) > 0:
                    dim_x = int(dim_x)
                    if len(dim_y) > 0:
                        dim_y = int(dim_y)
                        #st.write(int(dim_x) + int(dim_y))
                        #st.write(principalDf.shape)
                        #st.write(type(principalDf))
                        #st.write(principalDf.T[1].shape)
                        #st.write(type(dim_x))
                        img = principalDf.T[int(PC)].reshape((dim_x,dim_y))
                        if st.button("PC Hyperspectral image"):
                            fig5_0, ax5_0 = plt.subplots()
                            ax5_0.imshow(img)
                            ax5_0.set_title('PC' + str(PC) + ' Loading plot')
                            ax5_0.grid(False)

                            st.write(fig5_0) 


                loadings = pca.components_.T
                Loadings = pd.DataFrame(loadings)
                    
                PC_loading = np.array(Loadings.iloc[:,int(PC)])

                positive_lim1 = round(float(max(abs(PC_loading))*1.1),2) 
                negative_lim1 = round(float(max(abs(PC_loading))*-1.1),2) 

                

                if st.button("PC Loading plot"):
                    fig5_1, ax5_1 = plt.subplots()
                    ax5_1.plot(PC_loading) 
                    ax5_1.axhline(y = 0, color = 'k', linestyle = '--', alpha=0.5)
                    ax5_1.set_title('PC' + str(PC) + ' Loading plot')
                    ax5_1.set_ylim(negative_lim1,positive_lim1)
                    ax5_1.grid()

                    st.write(fig5_1)    


page_names_to_funcs = {  

    "Home Page": home_page,
    "Page 1: Unifeature FCS": page1,
    "Page 2: Bifeature FCS": page2, 
    "Page 3: (K-Means elbow plot)": page3,
    "Page 4: K-means directed shading (part 1))": page4,
    "Page 5: K-means directed shading (part 2))": page5,
    "Page 6: K-means directed shading (part 3))": page6,
    "Page 7 (PCA hyperspectral imaging)": page7,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()