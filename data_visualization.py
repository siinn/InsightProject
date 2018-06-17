#!/Users/sche/anaconda/bin/python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# set style sheet
plt.style.use("ggplot")
sns.set_style("white")


#------------------------------------------------
# start of main function
#------------------------------------------------
if __name__ == "__main__":


    # import cleaned dataset
    df = pd.read_csv("data/ny_rental_data_cleaned.r3.csv", index_col=0)
    
    # Describing numerical features
    df.describe()
    
    #------------------------------------------------
    # Number of bedrooms
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    
    # plot bedroom counts
    sns.countplot("bedrooms", data=df, ax=ax, hue="borough")
    
    # customize plots
    ax.set_xlabel("bedroom")
    ax.set_ylabel("Apartment")
    plt.legend(loc=1)

    # save figure
    plt.savefig("output/expl_bedrooms.pdf")
    
    
    #------------------------------------------------
    # Boroughs
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8, 5))
    
    # plot bedroom counts
    sns.countplot("borough", data=df, ax=ax)
    
    # customize plots
    ax.set_ylabel("Apartment")
    
    # save figure
    plt.savefig("output/expl_borough.pdf")
    
    #------------------------------------------------
    # Relationship between rent and bedrooms
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    
    # create bar plot
    sns.barplot(x="bedrooms",y="rent", data=df, ax=ax, hue="borough")
    
    # customize plots
    ax.legend()
    ax.set_ylabel("Rent ($)")
    ax.set_ylim([0,15000])
    
    # save figure
    plt.savefig("output/expl_rent_bedroom.pdf")
    
    #------------------------------------------------
    # Relationship between rent and apartment rating
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    
    # create bar plot
    sns.barplot(x="rating",y="rent", data=df, ax=ax, hue="borough")
    
    # customize plots
    ax.legend()
    ax.set_ylabel("Rent ($)")
    plt.legend(loc=2)
    
    # save figure
    plt.savefig("output/expl_rent_rating.pdf")
    
    #------------------------------------------------
    # Pet policy
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    # create bar plot
    sns.barplot(x="pet_dog",y="rent", data=df, ax=ax[0])
    sns.barplot(x="pet_cat",y="rent", data=df, ax=ax[1])
    
    # customize plots
    for a in ax:
        a.set_ylabel("Rent ($)")
        
    ax[0].set_xlabel("Dog allowed")
    ax[1].set_xlabel("Cat allowed")

    # save figure
    plt.savefig("output/expl_petpolicy.pdf")
    
    
    #------------------------------------------------
    # Amenities
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(3,3,figsize=(15,12))
    
    # create bar plot
    sns.barplot(x="amenity_washer/dryer",y="rent", data=df, ax=ax[0,0])
    sns.barplot(x="amenity_business",y="rent", data=df, ax=ax[0,1])
    sns.barplot(x="amenity_fitness",y="rent", data=df, ax=ax[0,2])
    sns.barplot(x="amenity_furnished",y="rent", data=df, ax=ax[1,0])
    sns.barplot(x="amenity_gated",y="rent", data=df, ax=ax[1,1])
    sns.barplot(x="amenity_pet care",y="rent", data=df, ax=ax[1,2])
    sns.barplot(x="amenity_yoga studio",y="rent", data=df, ax=ax[2,0])
    sns.barplot(x="amenity_package",y="rent", data=df, ax=ax[2,1])
    sns.barplot(x="amenity_concierge",y="rent", data=df, ax=ax[2,2])
    
    # customize plots
    for a in ax.reshape(-1):
        a.set_ylabel("Rent ($)")
        
    ax[0,0].set_xlabel("Washer/drier in unit")
    ax[0,1].set_xlabel("Business center")
    ax[0,2].set_xlabel("Fitness center")
    ax[1,0].set_xlabel("Furnished")
    ax[1,1].set_xlabel("Gated")
    ax[1,2].set_xlabel("Offer pet care")
    ax[2,0].set_xlabel("Yoga studio")
    ax[2,1].set_xlabel("Package service")
    ax[2,2].set_xlabel("Concierge")
    

    # save figure
    plt.savefig("output/expl_amenity.pdf")
    

    #------------------------------------------------
    # Relationship between rent and apartment size (sqft)
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(5,1,figsize=(15,8))
    
    ax[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
    ax[4] = plt.subplot2grid((2,6), (1,3), colspan=2)
    
    ax_count = 0
    # create regression plot (scatter)
    for borough, df_borough in df.groupby("borough"):
        sns.regplot("sqft", "rent", df_borough, fit_reg=False,
                    label=borough, scatter_kws={'alpha':0.8, 's':20}, ax=ax[ax_count])
        ax_count = ax_count+1
    
    # customize plots
    for a in ax.reshape(-1):
        a.set_ylabel("Rent ($)")
        a.set_ylim([0,25000])
        a.legend()
        
    # use tight layout
    plt.tight_layout()
    
    # save figure
    plt.savefig("output/expl_rent_sqft.pdf")
    

    #------------------------------------------------
    # Relationship between rent and built year
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(5,1,figsize=(15,8))
    
    ax[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
    ax[4] = plt.subplot2grid((2,6), (1,3), colspan=2)
    
    ax_count = 0
    # create regression plot (scatter)
    for borough, df_borough in df.groupby("borough"):
        sns.regplot("built_year", "rent", df_borough, fit_reg=False,
                    label=borough, scatter_kws={'alpha':0.8, 's':20}, ax=ax[ax_count])
        ax_count = ax_count+1
    
    # customize plots
    for a in ax.reshape(-1):
        a.set_ylabel("Rent ($)")
        a.set_ylim([0,25000])
        a.legend()
        
    # use tight layout
    plt.tight_layout()
    
    # save figure
    plt.savefig("output/expl_rent_builtYear.pdf")
    
    
    #------------------------------------------------
    # Relationship between rent and property size
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(5,1,figsize=(15,8))
    
    ax[0] = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
    ax[1] = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax[2] = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax[3] = plt.subplot2grid((2,6), (1,1), colspan=2)
    ax[4] = plt.subplot2grid((2,6), (1,3), colspan=2)
    
    ax_count = 0
    # create regression plot (scatter)
    for borough, df_borough in df.groupby("borough"):
        sns.regplot("property_size", "rent", df_borough, fit_reg=False,
                    label=borough, scatter_kws={'alpha':0.8, 's':20}, ax=ax[ax_count])
        ax_count = ax_count+1
    
    # customize plots
    for a in ax.reshape(-1):
        a.set_ylabel("Rent ($)")
        a.set_ylim([0,25000])
        a.legend()
        
    # use tight layout
    plt.tight_layout()
    
    # save figure
    plt.savefig("output/expl_rent_complex_size.pdf")
    
    #------------------------------------------------
    # Location
    #------------------------------------------------
    
    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(8,5))
    
    # create regression plot (scatter)
    for borough, df_borough in df.groupby("borough"):
        sns.regplot("longitude", "latitude", df_borough, fit_reg=False, scatter_kws={'s':10}, ax=ax, label=borough)
    
    # customize plots
    ax.legend()
    plt.tight_layout()
    
    # save figure
    plt.savefig("output/expl_long_lat.pdf")
    
    
    #------------------------------------------------
    # Heat map
    #------------------------------------------------

    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    
    # import gmaps to make scatter plot on google maps
    import gmaps, os
    gmaps.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    # data frame with coordinates only for easy plotting
    df_loc = df[["latitude","longitude"]]
    #df_loc = df_loc.dropna()
    
    # create heat layer
    layer = gmaps.heatmap_layer(df_loc, max_intensity=5, point_radius=8)
    
    # draw google map and the heat layer
    fig = gmaps.figure()
    fig.add_layer(layer)
    
    # save figure
    #plt.savefig("output/expl_heatmap.pdf")
    
    #------------------------------------------------
    # Location with Google map overlaid
    #------------------------------------------------

    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(10,8))
    
    # create figure
    fig = gmaps.figure()
    
    # define colors for each borough
    colors = [
              "rgba(200,112,126,0.7)", # red
              "rgba(76,146,177,0.7)", # blue
              "rgba(172,153,193,0.7)", # purple
              "rgba(200,194,189,0.7)", # gray
              "rgba(255,206,0,0.7)", # orange
              ]
    
    # loop over each boroguh by groupby
    color_index = 0
    for borough, df_borough in df.groupby("borough"):
        
        # remove rows with empty longitude and latitude. This should not happen when data cleaning is properly performed
        #df_borough = df_borough.dropna(subset=[["longitude","latitude"]])
        
        # only select latitude and logitude columns for easy plotting
        df_borough = df_borough[["latitude","longitude"]]
      
        # set color to current index
        color = colors[color_index]
        color_index = color_index+1
    
        # create symbol layer, representing apartment listings from each borough
        symbol_layer = gmaps.symbol_layer(df_borough,
                                          fill_color=color,
                                          stroke_color=color,
                                          scale=2)
        # add symbol_lyaer to figure
        fig.add_layer(symbol_layer)
    
    
    # save figure
    #plt.savefig("output/expl_overlay.pdf")
    
    
    #------------------------------------------------
    # Correlation of continuous features
    #------------------------------------------------

    # set subplots
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    
    # select columns with numeric values
    df_lin = df[["bathrooms","bedrooms","leaseLength","rating","sqft","built_year","property_size","longitude","latitude"]]
    
    # TEMPORARY FIX: drop rows without longitude and latitude
    #df_lin = df_lin.dropna(subset=["longitude","latitude"])

    # make a pairplot
    sns.pairplot(df_lin)
    
    # save figure
    plt.savefig("output/expl_correlation.pdf")
