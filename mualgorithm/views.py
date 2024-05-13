from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

def hi(request):
    return render(request,'index.html')
import matplotlib.pyplot as plt
from django.http import HttpResponse
import io,os
# views.py
import numpy as np
from django.shortcuts import render
import matplotlib.pyplot as plt
import io
# views.py
from django.shortcuts import render
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
import csv
import pandas as pd

from scipy.spatial.distance import cdist

from imblearn.over_sampling import SMOTE,ADASYN
def summary(request): 
    # Read CSV file into DataFrame
    csv_file = request.FILES['csv_file']
    df = pd.read_csv(csv_file)

    X = df.drop(columns=['y']).values
    y = df['y'].values

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Majority class', alpha=0.5)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Minority class', alpha=0.5)
    plt.title('Original Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    # Convert plot to base64 encoded string
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    orginal_dataset = base64.b64encode(img_data.getvalue()).decode()

    # Clear plot to avoid memory leaks
    plt.clf()

    # Pass base64 encoded plot to template
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)
    
    
    # Plot SMOTE Resampled Data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_resampled_smote[y_resampled_smote == 0][:, 0], X_resampled_smote[y_resampled_smote == 0][:, 1], label='Majority class', alpha=0.5)
    plt.scatter(X_resampled_smote[y_resampled_smote == 1][:, 0], X_resampled_smote[y_resampled_smote == 1][:, 1], label='Minority class', alpha=0.5)
    plt.title('SMOTE Resampled Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    
    
    # Plot SMOTE Resampled Data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_resampled_smote[y_resampled_smote == 0][:, 0], X_resampled_smote[y_resampled_smote == 0][:, 1], label='Majority class', alpha=0.5)
    plt.scatter(X_resampled_smote[y_resampled_smote == 1][:, 0], X_resampled_smote[y_resampled_smote == 1][:, 1], label='Minority class', alpha=0.5)
    plt.title('SMOTE Resampled Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    # Convert plot to base64 encoded string
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    smote_dataset = base64.b64encode(img_data.getvalue()).decode()

    # Clear plot to avoid memory leaks
    plt.clf()


    X_minority = df[df['y'] == df['y'].value_counts().idxmin()].values

    X_majority = df[df['y'] == df['y'].value_counts().idxmax()].values

    # Compute pairwise distances between all points in the majority class and minority class
    distances = cdist(X_majority, X_minority)

    # Find the indices of the pair with the maximum distance
    majority_index, minority_index = np.unravel_index(np.argmax(distances), distances.shape)

    # Get the points with maximum distance
    point_majority = X_majority[majority_index]
    point_minority = X_minority[minority_index]

    print("Point from majority class:", point_majority)
    print("Point from minority class:", point_minority)




    # Compute distances between point_minority and all points in the majority class
    distances_to_minority = np.linalg.norm(X_majority - point_minority, axis=1)

    # Find the index of the point in the majority class closest to point_minority
    closest_majority_index = np.argmin(distances_to_minority)

    # Get the point from the majority class closest to point_minority
    closest_majority_point = X_majority[closest_majority_index]

    print("Point from the majority class closest to point_minority:", closest_majority_point)






    # Compute distances between point_majority and all points in the minority class
    distances_to_majority = np.linalg.norm(X_minority - point_majority, axis=1)

    # Find the index of the point in the minority class closest to point_majority
    closest_minority_index = np.argmin(distances_to_majority)

    # Get the closest point from the minority class
    closest_minority_point = X_minority[closest_minority_index]

    print("Closest point from the minority class to point_majority:", closest_minority_point)
















    # Calculate the radius of the circle
    radius = np.linalg.norm(closest_minority_point - closest_majority_point)

    # Calculate the distances of each minority point to the center of the circle
    distances_to_center = np.linalg.norm(X_minority - closest_minority_point, axis=1)

    # Filter out the minority points lying within the circle
    filtered_minority_points = X_minority[distances_to_center > radius]

    print("Number of minority points before filtering:", len(X_minority))
    print("Number of minority points after filtering:", len(filtered_minority_points))





    # Plot Original Data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Majority class', alpha=0.5)
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='Minority class', alpha=0.5)
    plt.title('First Circle')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    # Plot the circle
    circle = plt.Circle((closest_minority_point[0], closest_minority_point[1]), radius, color='r', fill=False)
    plt.gca().add_patch(circle)




    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    first_circle = base64.b64encode(img_data.getvalue()).decode()

    # Clear plot to avoid memory leaks
    plt.clf()



    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Majority class', alpha=0.5)
    plt.scatter(filtered_minority_points[:, 0], filtered_minority_points[:, 1], label='Minority class (filtered)', alpha=0.5)
    plt.title('Filtered dataset after filtering from first circle')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    # Plot the circle
    circle = plt.Circle((closest_minority_point[0], closest_minority_point[1]), radius, color='r', fill=False)
    plt.gca().add_patch(circle)

    
    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    filtered_data_by_first_circle = base64.b64encode(img_data.getvalue()).decode()

    # Clear plot to avoid memory leaks
    plt.clf()


# Plot Original Data
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='Majority class', alpha=0.5)
    plt.scatter(filtered_minority_points[:, 0], filtered_minority_points[:, 1], label='Minority class (Filtered)', alpha=0.5)
    plt.title('Second Circle')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    minority_del_num=len(filtered_minority_points)-len(X_minority);
    minority_del=filtered_minority_points;



    # Plot the circle
    circle = plt.Circle((closest_majority_point[0], closest_majority_point[1]), radius, color='r', fill=False)
    plt.gca().add_patch(circle)

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    second_circle = base64.b64encode(img_data.getvalue()).decode()

    # Clear plot to avoid memory leaks
    plt.clf()




    

    # Calculate the distances of each majority point to the center of the circle
    distances_to_center_majority = np.linalg.norm(X_majority - closest_majority_point, axis=1)

    # Filter out the majority points lying within the circle
    filtered_majority_points = X_majority[distances_to_center_majority > radius]

    print("Number of majority points before filtering:", len(X_majority))
    print("Number of majority points after filtering:", len(filtered_majority_points))












    # Plot Original Data
    plt.figure(figsize=(8, 6))
    plt.scatter(filtered_majority_points[:, 0], filtered_majority_points[:, 1], label='Majority class (Filtered)', alpha=0.5)
    plt.scatter(filtered_minority_points[:, 0], filtered_minority_points[:, 1], label='Minority class (Filtered)', alpha=0.5)
    plt.title('Filtered dataset after filtering from Second circle')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # Plot the circle
    circle = plt.Circle((closest_majority_point[0], closest_majority_point[1]), radius, color='r', fill=False)
    plt.gca().add_patch(circle)

    plt.legend()
    plt.grid(True)

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    filtered_data_by_second_circle = base64.b64encode(img_data.getvalue()).decode()

    # Clear plot to avoid memory leaks
    plt.clf()


    majority_del_num=len(filtered_majority_points)-len(X_majority);
    majority_del=filtered_majority_points;






    # Apply SMOTE on the remaining dataset
    smote_remaining = SMOTE(random_state=42)
    X_resampled_smote_remaining, y_resampled_smote_remaining = smote_remaining.fit_resample(np.vstack([filtered_majority_points, filtered_minority_points]), np.hstack([np.zeros(len(filtered_majority_points)), np.ones(len(filtered_minority_points))]))

    # Plot SMOTE Resampled Data with filtered majority and minority points
    plt.figure(figsize=(8, 6))
    plt.scatter(X_resampled_smote_remaining[y_resampled_smote_remaining == 0][:, 0], X_resampled_smote_remaining[y_resampled_smote_remaining == 0][:, 1], label='Majority class (SMOTE)', alpha=0.5)
    plt.scatter(X_resampled_smote_remaining[y_resampled_smote_remaining == 1][:, 0], X_resampled_smote_remaining[y_resampled_smote_remaining == 1][:, 1], label='Minority class (SMOTE)', alpha=0.5)
    plt.title('Smote applied dataset after filtering from both circle')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)

    img_data = io.BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    smote_after_filtering = base64.b64encode(img_data.getvalue()).decode()

    # Clear plot to avoid memory leaks
    plt.clf()


    
    context = {'majority_del_num':majority_del_num,'majority_del':majority_del,'minority_del_num':minority_del_num,'minority_del':minority_del,'smote_after_filtering':smote_after_filtering,'filtered_data_by_second_circle':filtered_data_by_second_circle,'filtered_data_by_first_circle':filtered_data_by_first_circle,'second_circle':second_circle,'first_circle':first_circle,'orginal_dataset': orginal_dataset,'smote_dataset': smote_dataset}
        
    # Render template with plot
    return render(request, 'index.html', context)


