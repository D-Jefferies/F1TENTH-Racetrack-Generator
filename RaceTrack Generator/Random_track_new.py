import cv2
import os
import math
import numpy as np
import shapely.geometry as shp
import matplotlib.pyplot as plt
from sympy.geometry import Point, Polygon
from matplotlib.collections import PatchCollection
from shapely.geometry import MultiPoint
import argparse
import time 
from scipy.interpolate import splprep, splev
from scipy.interpolate import PchipInterpolator
import random
parser = argparse.ArgumentParser()
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

# parser.add_argument('--seed', type=int, default=1245, help='Seed for the numpy rng.')
parser.add_argument('--num_maps', type=int, default=10, help='Number of maps to generate.')
args = parser.parse_args()

# np.random.seed(args.seed)
current_time = int(time.time())

# Set the random seed based on the current time
np.random.seed(current_time)
# np.random.seed(1455)

same_size = False

if not os.path.exists('maps'):
    print('Creating maps/ directory.')
    os.makedirs('maps')
if not os.path.exists('centerline'):
    print('Creating centerline/ directory.')
    os.makedirs('centerline')

NUM_MAPS = args.num_maps
WIDTH = 3.0
LENGTH = 120
a = 20
b = 30
h = 10
k = 15
y = []
# x_val= np.linspace(0,20,num = 30)

def plot_ellipse(center_x, center_y, semi_major_axis, semi_minor_axis, angle_degrees, num_points=1000):
    # Convert angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Generate angles for the ellipse points
    angles = np.linspace(0, 2 * np.pi, num_points)

    # Calculate x and y coordinates for the ellipse points
    x_coords = center_x + semi_major_axis * np.cos(angles) * np.cos(angle_radians) - semi_minor_axis * np.sin(angles) * np.sin(angle_radians)
    y_coords = center_y + semi_major_axis * np.cos(angles) * np.sin(angle_radians) + semi_minor_axis * np.sin(angles) * np.cos(angle_radians)

    # Plot the ellipse
    # plt.plot(x_coords, y_coords)
    # plt.show()
    return x_coords,y_coords

def calculate_rotated_rectangle_perimeter(center_x, center_y, width, height, num_points, rotation_angle_deg):
    rotation_angle_rad = math.radians(rotation_angle_deg)

    x_min = center_x - width / 2
    x_max = center_x + width / 2
    y_min = center_y - height / 2
    y_max = center_y + height / 2

    perimeter_coords = []
    for x in np.linspace(x_min, x_max, num_points):
        rotated_x = math.cos(rotation_angle_rad) * (x - center_x) - math.sin(rotation_angle_rad) * (y_min - center_y) + center_x
        rotated_y = math.sin(rotation_angle_rad) * (x - center_x) + math.cos(rotation_angle_rad) * (y_min - center_y) + center_y
        perimeter_coords.append((rotated_x, rotated_y))
    
    for y in np.linspace(y_min, y_max, num_points):
        rotated_x = math.cos(rotation_angle_rad) * (x_max - center_x) - math.sin(rotation_angle_rad) * (y - center_y) + center_x
        rotated_y = math.sin(rotation_angle_rad) * (x_max - center_x) + math.cos(rotation_angle_rad) * (y - center_y) + center_y
        perimeter_coords.append((rotated_x, rotated_y))
    
    for x in np.linspace(x_max, x_min, num_points):
        rotated_x = math.cos(rotation_angle_rad) * (x - center_x) - math.sin(rotation_angle_rad) * (y_max - center_y) + center_x
        rotated_y = math.sin(rotation_angle_rad) * (x - center_x) + math.cos(rotation_angle_rad) * (y_max - center_y) + center_y
        perimeter_coords.append((rotated_x, rotated_y))
    
    for y in np.linspace(y_max, y_min, num_points):
        rotated_x = math.cos(rotation_angle_rad) * (x_min - center_x) - math.sin(rotation_angle_rad) * (y - center_y) + center_x
        rotated_y = math.sin(rotation_angle_rad) * (x_min - center_x) + math.cos(rotation_angle_rad) * (y - center_y) + center_y
        perimeter_coords.append((rotated_x, rotated_y))

    x_coords, y_coords = zip(*perimeter_coords)

    return x_coords, y_coords


def calculate_rotated_pentagon_perimeter(center_x, center_y, side_length, rotation_angle_deg, num_points=1000):
    rotation_angle_rad = math.radians(rotation_angle_deg)
    
    pentagon_vertices = []
    for i in range(5):
        angle = i * (2 * math.pi / 5) + rotation_angle_rad
        x = center_x + side_length * math.cos(angle)
        y = center_y + side_length * math.sin(angle)
        pentagon_vertices.append((x, y))
    
    perimeter_coords = []
    for i in range(5):
        start_x, start_y = pentagon_vertices[i]
        end_x, end_y = pentagon_vertices[(i + 1) % 5]
        x_vals = np.linspace(start_x, end_x, num_points // 5)
        y_vals = np.linspace(start_y, end_y, num_points // 5)
        perimeter_coords.extend(zip(x_vals, y_vals))
    x_coords, y_coords = zip(*perimeter_coords)
    # plot_rotated_pentagon(x_coords, y_coords)
    return x_coords,y_coords

def plot_rotated_pentagon(xs,ys):

    plt.plot(xs, ys, color = 'purple')
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rotated Pentagon')
    plt.grid()
    plt.show()

def deform_and_create(x_coords,y_coords):

    x_deform, y_deform = deform_track(x_coords,y_coords)
    # Show the plot
    ######################################3

    Track_coords = [(x_deform,y_deform) for x_deform,y_deform in zip(x_deform, y_deform)]
    Track_coords = np.asarray(Track_coords)
    # Create Shapely Polygon objects for the ellipse and rectangle
    Track_poly = np.asarray(Track_coords)
    # Create a Shapely Polygon object for the shape
    Track_poly = shp.Polygon(Track_poly)
    # Check if the polygon is valid (i.e., it doesn't self-intersect)
    is_valid = Track_poly.is_valid
    if not is_valid:
        # Attempt to fix self-intersections
        valid_polygon = Track_poly.buffer(0)
        if valid_polygon.is_valid:
            if isinstance(valid_polygon, shp.MultiPolygon):
                for geom in valid_polygon.geoms:
                    if geom.is_valid:
                        valid_points = geom.exterior.coords
                        x_deform, y_deform = zip(*valid_points)
            else:
                valid_points = valid_polygon.exterior.coords
                x_deform, y_deform = zip(*valid_points)


    #########################################
    track = [(a1, b1, x, y) for a1, b1, x, y in zip(x_deform, y_deform, x_deform, y_deform)]
    track_xy = [(x, y) for (a1, b1, x, y) in track]
    track_xy = [(x, y) for x, y in zip(x_deform, y_deform)]

    if same_size == True:

        track_xy = resize_tack(track_xy)

    track_xy = place_on_origin(track_xy)

    track_xy = np.asarray(track_xy)
    track_poly = shp.Polygon(track_xy)

    # Offset the ellipse inward and outward
    WIDTH = 3  # Change this value to set the width of the offset
    track_xy_offset_in = track_poly.buffer(2*WIDTH)
    # track_xy_center= track_poly.buffer(WIDTH)
    # track_xy_offset_out = track_poly.buffer(0)
    track_xy_center = track_xy_offset_in.buffer(-WIDTH)
    track_xy_offset_out = track_xy_offset_in.buffer(-2*WIDTH)
    track_xy_offset_in_np = np.array(track_xy_offset_in.exterior.coords)
    track_xy_offset_out_np = np.array(track_xy_offset_out.exterior.coords)
    track_xy = np.array(track_xy_center.exterior.coords)
    plt.figure()
    # Plot the offset polygons
    plt.plot(track_xy[:,0],track_xy[:,1],linestyle='--')
    plt.plot(track_xy_offset_in_np[:, 0], track_xy_offset_in_np[:, 1] ,linestyle='-')
    plt.plot(track_xy_offset_out_np[:, 0], track_xy_offset_out_np[:, 1], linestyle='-')
    plt.axis("equal")
    plt.xticks([])
    plt.yticks([])
    plt.savefig('maps/Track_done' + '_map.pdf', dpi=580)


    # Add a legend to differentiate between the ellipse and offset polygons
    #plt.legend()

    # plt.show()
    plt.xticks([])
    plt.yticks([])   
    plt.figure()
    return track_xy,track_xy_offset_in_np,track_xy_offset_out_np

def find_outline_coords(shape1_x_coords, shape1_y_coords, shape2_x_coords, shape2_y_coords):
    plt.plot(shape1_x_coords,shape1_y_coords,linestyle = '-', color = "blue")
    plt.plot(shape2_x_coords,shape2_y_coords,linestyle = '-', color = "red")
    plt.xticks([])
    plt.yticks([])
    plt.savefig('maps/Track_shapes' + '_map.pdf', dpi=580)
    # plt.show()
    
    shape1_coords = [(x,y) for x,y in zip(shape1_x_coords, shape1_y_coords)]
    shape2_coords = [(x,y) for x,y in zip(shape2_x_coords, shape2_y_coords)]
    shape1_coords = np.asarray(shape1_coords)
    shape2_coords = np.asarray(shape2_coords)

    # Create Shapely Polygon objects for the ellipse and rectangle
    shape1_polygon = shp.Polygon(shape1_coords)
    shape2_polygon = shp.Polygon(shape2_coords)

    # Find the intersection points
    intersection_points = shape1_polygon.union(shape2_polygon)
    intersection_points_arr = np.array(intersection_points.exterior.coords)
    x_coords = intersection_points_arr[:,0]
    y_coords = intersection_points_arr[:,1]
    plt.plot(x_coords,y_coords,color = 'green')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('maps/Track_intersect' + '_map.pdf', dpi=580)
    # plt.show()
    return x_coords,y_coords


def deform_track(x_coords, y_coords):
    x_deform =[]
    y_deform = []
    done = False
    deform_index_start = 2
    connect_hairpin = False
    connect_feature = False
    connect_chicane = False
    while not done:
        deform_index_end = deform_index_start + len(x_coords)/4
        next_deform = np.random.randint(deform_index_start,deform_index_end)

        if next_deform >= len(x_coords)-2:
            next_deform = len(x_coords)-2
            done = True

        x_start = x_coords[deform_index_start]
        y_start = y_coords[deform_index_start]
        x_end = x_coords[next_deform]
        y_end = y_coords[next_deform]

        num_points =  next_deform-deform_index_start
        poly_weight = 0.35
        feature_weigth = 0.075
        chicane_weight = 0.0
        straight_weigth = 0.3
        hairpin_weight = 0.075
        shape_weight = 0.15



        if num_points > 300:
            # weights = [0.35, 0.015, 0.0, 0.2,0.01, 0.15] 
            feature_weigth = 0.1
            hairpin_weight = 0.1 
            chicane_weight = 0.0
            poly_weight = 0.4
            shape_weight = 0.25
            straight_weigth = 0.15

        elif num_points < 100:
            # weights = [0.375, 0.0, 0.25, 0.225,0.0, 0.125]
            feature_weigth = 0
            hairpin_weight = 0   
            chicane_weight = 0.0
            poly_weight = 0.5
            shape_weight = 0.4
            straight_weigth = 0.25

        if connect_feature or connect_hairpin:
            feature_weigth = 0
            hairpin_weight = 0   
            chicane_weight = 0.0
            poly_weight = 0.475
            shape_weight = 0.155
            straight_weigth = 0.35

        if connect_chicane:
            poly_weight = 0.5
            feature_weigth = 0.1
            chicane_weight = 0.00
            straight_weigth = 0.2
            hairpin_weight = 0.1
            shape_weight = 0.15     

        if LENGTH < 300 and same_size: 
            feature_weigth = 0
            hairpin_weight = 0   
            chicane_weight = 0.0
            poly_weight = 0.475
            shape_weight = 0.125
            straight_weigth = 0.30      

        options = [1, 2, 3, 4, 5, 6]
        weights = [poly_weight, feature_weigth, chicane_weight, straight_weigth,hairpin_weight,shape_weight]  # Adjust the weights to your desired bias

        selected_option = random.choices(options, weights=weights, k=1)[0]
        if selected_option == 1:
            connect_poly = True 
            connect_feature= False
            connect_chicane = False
            connect_straight = False
            connect_hairpin = False
            connect_shape = False
        if selected_option == 2:
            connect_poly = False 
            connect_feature= True
            connect_chicane = False
            connect_straight = False
            connect_hairpin = False
            connect_shape = False
        if selected_option == 3:
            connect_poly = False 
            connect_feature= False
            connect_chicane = True
            connect_straight = False
            connect_hairpin = False
            connect_shape = False
        if selected_option == 4:
            connect_poly = False 
            connect_feature= False
            connect_chicane = False
            connect_straight = True
            connect_hairpin = False
            connect_shape = False
        if selected_option == 5:
            connect_poly = False 
            connect_feature= False
            connect_chicane = False
            connect_straight = False
            connect_hairpin = True       
            connect_shape = False   
        if selected_option == 6:
            connect_poly = False 
            connect_feature= False
            connect_chicane = False
            connect_straight = False
            connect_hairpin = False 
            connect_shape = True 

################################################################
#fit with curve or fit poly 
        
        if connect_poly:
            x_values,y_values = connect_with_poly(x_start,y_start,x_end,y_end,num_points)
  
        if connect_chicane:
            mid_point_index = deform_index_start + round(num_points/2)
            x_mid = x_coords[mid_point_index]
            y_mid = y_coords[mid_point_index]
            x_values,y_values = connect_with_chicane(x_start,y_start,x_mid,y_mid,x_end,y_end,num_points)
        
        if connect_straight:
            x_values,y_values = connect_with_straight(x_start,y_start,x_end,y_end,num_points)
        
        if connect_feature:
            x_values,y_values = connect_with_feature(x_start,y_start,x_end,y_end,num_points)
        
        if connect_hairpin:
            x_values,y_values = connect_with_hairpin(x_start,y_start,x_end,y_end,num_points)
        
        if connect_shape:
            x_values,y_values = Connect_with_shape(x_coords,y_coords,deform_index_start,next_deform)
            

################################################################
        x_deform.extend(x_values)
        y_deform.extend(y_values)
        plt.figure()
        plt.plot(x_coords, y_coords)
        plt.plot(x_deform, y_deform,color = 'red')
        plt.plot(x_values, y_values,color = 'green')

        deform_index_start = next_deform + 1

        plt.plot(x_deform, y_deform,color = 'red')
        plt.xticks([])
        plt.yticks([])
        # plt.savefig('maps/Track_deform' + str(deform_index_start)+'_map.pdf', dpi=580)
        # plt.show()

    try:
        tck, u = splprep([x_deform, y_deform], s = 60)
        u_new = np.linspace(0, 1, num=1000)
        x_spline_2, y_spline_2 = splev(u_new, tck)
        # distances = np.sqrt(np.diff(x_spline_2)**2 + np.diff(y_spline_2)**2)
        plt.plot(x_spline_2,y_spline_2,color = 'black')
        plt.xticks([])
        plt.yticks([])
        # plt.savefig('maps/Track_spline' + '_map.pdf', dpi=580)


    except:
        print("Error !")
    return x_spline_2,y_spline_2


def Connect_with_shape(x_coords,y_coords,deform_index_start,next_deform):
    x_shape = x_coords[deform_index_start:next_deform]
    y_shape = y_coords[deform_index_start:next_deform]
    # plt.plot(x_shape,y_shape,color = "black")
    # plt.show()
    return x_shape,y_shape

def connect_with_feature(x_start,y_start,x_end,y_end,num_points):
    x_values =[]
    y_values =[]
    x_values.append(x_start)
    y_values.append(y_start)
    dx = x_end-x_start
    dy = y_end - y_start
    angle = np.arctan2(dy, dx)
    angle_new = np.arctan2(dy, dx) + math.pi / 2
    distance = np.random.randint(5,50)
    radius = np.random.randint(5,25)
    point_x = x_end + distance * math.cos(angle_new)
    point_y = y_end + distance * math.sin(angle_new)
    angles = (angle_new + math.pi/2) - np.linspace(0,math.pi, (num_points-4))
    x_coords = point_x + radius * np.cos(angles)
    y_coords = point_y + radius * np.sin(angles)
    choice = np.random.randint(0,10)
    if choice > 5:
        x_values.append((x_start+point_x)/2)
        y_values.append((y_start+point_y)/2)
    else: 
        x_values.append((x_coords[0] + x_start)/2)
        y_values.append((y_coords[0] + y_start)/2)

    x_values.extend(x_coords)
    y_values.extend(y_coords)
    x_values.append((x_coords[-1] + x_end)/2)
    y_values.append((y_coords[-1] + y_end)/2)   
    x_values.append(x_end)
    y_values.append(y_end)
    # plt.plot(x_values,y_values)
    # plt.show()
    return x_values, y_values 

def connect_with_hairpin(x_start,y_start,x_end,y_end,num_points):
    x_values =[]
    y_values =[]
    dx = x_end - x_start
    dy = y_end - y_start
    div = np.random.randint(3,7)
    angle = np.arctan2(dy, dx) + math.pi / div
    distance = np.random.randint(5,35)
    point_x = x_start + distance * math.cos(angle)
    point_y = y_start + distance * math.sin(angle)
    slope = (point_y - y_start) / (point_x - x_start)
    intercept = y_start - slope * x_start
    x_coords = np.linspace(x_start,(point_x - 0.01),round(num_points/2 - 1)) 
    y_coords = slope*x_coords + intercept
    x_values.extend(x_coords)
    y_values.extend(y_coords)
    # x_coords = point_x + 1
    # y_coords = -slope*x_coords + intercept
    slope = (point_y - y_end) / (point_x - x_end)
    intercept = y_end - slope * x_end
    x_coords = np.linspace(point_x,x_end,round(num_points/2)) 
    y_coords = slope*x_coords + intercept
    x_values.extend(x_coords)
    y_values.extend(y_coords)
    plt.plot(x_values,y_values)
    # plt.show()
    return x_values, y_values     

def connect_with_poly(x_start,y_start,x_end,y_end,num_points):
    x_poly = np.array([x_start, x_end])
    y_poly = np.array([y_start, y_end])
    degree = np.random.randint(3,5)
    coefficients = np.polyfit(x_poly, y_poly, degree)
    polynomial_func = np.poly1d(coefficients)
    x_values = np.linspace(x_start, x_end, num_points)
    y_values = polynomial_func(x_values)
    return x_values, y_values 

def connect_with_chicane(x_start,y_start,x_mid,y_mid,x_end,y_end,num_points):
    plt.plot(x_mid,y_mid,marker = 'x',color = 'grey')
    x_poly = np.array([x_start,x_mid,x_end])
    y_poly = np.array([y_start,y_mid,y_end])
    degree = 3
    coefficients = np.polyfit(x_poly, y_poly, degree)
    polynomial_func = np.poly1d(coefficients)
    x_values = np.linspace(x_start, x_end, num_points)
    y_values = polynomial_func(x_values)
    # plt.plot(x_values,y_values,color = "red")
    # plt.show()
    return x_values, y_values 

def connect_with_straight(x_start,y_start,x_end,y_end,num_points):

    x_values = np.linspace(x_start, x_end, num_points)
    grad = (y_end-y_start)/(x_end-x_start)
    c = y_start-grad*x_start
    y_values = x_values*grad +c
    return x_values, y_values

def convert_track(track, track_int, track_ext, iter):

    # converts track to image and saves the centerline as waypoints
    fig, ax = plt.subplots()
    fig.set_size_inches(50,50)
    # ax.plot(track[:,0],track[:,1], color = "black", linewidth=2,linestyle = "--")
    # ax.plot(track[0:348,0],track[0:348,1], color = "red", linewidth=2,linestyle = "--")
    # ax.plot(track[0, 0], track[0, 1], 'ro', markersize=20)
    # ax.plot(track[0, 0], track[0, 1], marker='o', markersize=20, markerfacecolor='red', markeredgecolor='red', linestyle='None')
    ax.plot(*track_int.T, color='black', linewidth = 3)
    ax.plot(*track_ext.T, color='black', linewidth = 3)
    plt.tight_layout()
    ax.set_aspect('equal')
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    plt.axis('off')
    plt.savefig('maps/Track' + str(iter) + '_map.png', dpi=80)
    # plt.savefig('maps/Track' + str(iter) + '_map_color.pdf', dpi=580)

    # plt.show()
    # print(*track_int.T)
    # print('******************************************************************')
    # print(*track_ext.T)
    map_width, map_height = fig.canvas.get_width_height()
    print('map size: ', map_width, map_height)

    # transform the track center line into pixel coordinates
    xy_pixels = ax.transData.transform(track)
    # map_origin_x = 0
    # map_origin_y = 0

    
    origin_x_pix = xy_pixels[0, 0]
    origin_y_pix = xy_pixels[0, 1]

    xy_pixels = xy_pixels - np.array([[origin_x_pix, origin_y_pix]])

    map_origin_x = -origin_x_pix*0.05
    map_origin_y = -origin_y_pix*0.05

    # convert image using cv2
    cv_img = cv2.imread('maps/Track' + str(iter) + '_map.png', -1)
    # convert to bw
    cv_img_bw = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # saving to img
    cv2.imwrite('maps/Track' + str(iter) + '_map.png', cv_img_bw)
    # cv2.imwrite('maps/map' + str(iter) + '.pgm', cv_img_bw)

    # create yaml file
    yaml = open('maps/Track' + str(iter) + '_map.yaml', 'w')
    yaml.write('image: Track' + str(iter) + '_map.png\n')
    yaml.write('resolution: 0.062500\n')
    yaml.write('origin: [' + str(map_origin_x) + ',' + str(map_origin_y) + ', 0.000000]\n')
    yaml.write('negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196')
    yaml.close()
    plt.close()

    # saving track centerline as a csv in ros coords
    waypoints_csv = open('maps/Track' + str(iter) + '_centerline.csv', 'w')
    waypoints_csv.write('Xm,Ym,Wr,Wl\n')
    for row in xy_pixels[::-1]:
        waypoints_csv.write(str(0.05*row[0]) + ', ' + str(0.05*row[1]) + ',1.1,1.1\n')
    waypoints_csv.close()
    # time.sleep(3)

def resize_tack(xy_coords):
    track_len = calculate_track_length(xy_coords)
    # xy_coords = xy_coords * 450/track_len
    xy_coords = [(LENGTH/track_len * x, LENGTH/track_len * y) for x, y in xy_coords]
    track_len_check = calculate_track_length(xy_coords)

    return xy_coords




def calculate_distance(x1, y1, x2, y2):
    # Calculate the distance between two points using Euclidean distance formula
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def calculate_track_length(coordinates):
    track_length = 0

    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        distance = calculate_distance(x1, y1, x2, y2)
        track_length += distance

    return track_length

def place_on_origin(coordinates):
    x1, y1 = coordinates[1]
    min_x = abs(x1)
    min_y = abs(y1)
    for i in range(len(coordinates)):
        x1, y1 = coordinates[i]
        if abs(x1) < min_x and abs(y1) < min_y:
            min_x = x1
            min_y = y1
    xy_coords = [(x - min_x, y - min_y) for x, y in coordinates]
    return xy_coords



if __name__ == '__main__':
    
    for i in range(0,1):

        shape_1 = np.random.randint(0,3)
        shape_2 = np.random.randint(0,3)
        center_x_1 = 25
        center_y_1 = 15
        if shape_1 == 0:
            semi_major_axis = np.random.randint(30,150)
            semi_minor_axis = np.random.randint(10,round(semi_major_axis/2))
            angle_degrees = np.random.randint(0,180)
            shape_1_x, shape_1_y = plot_ellipse(center_x_1, center_y_1, semi_major_axis, semi_minor_axis, angle_degrees, num_points=1000)
        elif shape_1 == 1:
            center_x = center_x_1 + np.random.randint(-1*center_x_1, center_x_1)
            center_y = center_y_1 + np.random.randint(-1*center_y_1,center_y_1) 
            rotation_angle_deg =  np.random.randint(0,180)
            height = np.random.randint(30,150)
            width = np.random.randint(10,round(height/1.2))
            num_points = 250
            angle_degrees = np.random.randint(0,180)
            shape_1_x, shape_1_y = calculate_rotated_rectangle_perimeter(center_x, center_y, width, height, num_points,angle_degrees)
        elif shape_1 == 2:
            center_x = center_x_1 + np.random.randint(-1*center_x_1, center_x_1)
            center_y = center_y_1 + np.random.randint(-1*center_y_1,center_y_1)  
            side_length = np.random.randint(10,50) 
            rotation_angle_deg = np.random.randint(0,180)
            shape_1_x,shape_1_y = calculate_rotated_pentagon_perimeter(center_x, center_y, side_length, rotation_angle_deg,num_points = 1000)     



        if shape_2 == 0:
            center_x = center_x_1 + np.random.randint(-1*center_x_1,2*center_x_1)
            center_y = center_y_1 + np.random.randint(-1*center_y_1,2*center_y_1) 
            semi_major_axis = np.random.randint(30,150)
            semi_minor_axis = np.random.randint(10,round(semi_major_axis/2))           
            angle_degrees = np.random.randint(0,180)
            shape_2_x, shape_2_y = plot_ellipse(center_x, center_y, semi_major_axis, semi_minor_axis, angle_degrees, num_points=1000)
        elif shape_2 == 1:
            center_x = center_x_1 + np.random.randint(-1*center_x_1, 2*center_x_1)
            center_y = center_y_1 + np.random.randint(-1*center_y_1,2*center_y_1) 
            rotation_angle_deg = np.random.randint(0,180)
            height = np.random.randint(30,150)
            width = np.random.randint(10,round(height/1.2))
            num_points = 250
            angle_degrees = np.random.randint(0,180)
            shape_2_x, shape_2_y = calculate_rotated_rectangle_perimeter(center_x, center_y, width, height, num_points,angle_degrees)
        elif shape_2 == 2:
            center_x = center_x_1 + np.random.randint(-1*center_x_1, 2*center_x_1)
            center_y = center_y_1 + np.random.randint(-1*center_y_1,2*center_y_1)  
            side_length = np.random.randint(10,50) 
            rotation_angle_deg = np.random.randint(0,180)
            shape_2_x,shape_2_y = calculate_rotated_pentagon_perimeter(center_x, center_y, side_length, rotation_angle_deg,num_points = 1000)
        
        # plt.show()

        # outline_x_coords,outline_y_coordes = find_outline_coords(shape_1_x, shape_1_y, shape_2_x, shape_2_y)
        
        try:
            outline_x_coords,outline_y_coordes = find_outline_coords(shape_1_x, shape_1_y, shape_2_x, shape_2_y)
            track,track_int,track_ext = deform_and_create(outline_x_coords,outline_y_coordes)
            convert_track(track, track_int, track_ext, i)
        except:
            print("Error !!!")
    