############################
# 
#   lhsp_mom_viewer
#       Liam A. Taylor, June 2022
#       Update _v05 7/25/2022 by R. Mauck
#           Changes: make this a  function/prodedure based app as first step to make it object-oriented, then to GUI
#       Update _v05 7/26 by RAM
#           Encapsulated everyting until multiple birds in v04, this proceeds from there
#               have deleted all the code that was commented out in v04
#               you can always go look at v04 to see deleted
#           Encapsulated everything 
# 
# ##########

# Built with
#   Python 3.10.4
#   sys
#   pandas
#   matplotlib
#   numpy
#   statistics
#   scipy

import sys
import pandas
import matplotlib.pyplot as pyplot
import matplotlib.backend_bases as backend
from matplotlib.widgets import Button 
import numpy as np
import statistics
from scipy import stats 

################### 
#  declare some global variables
####
global user_INPATH
global user_BURROW
global data_DATE

global data ## for dataframes to be defined later
global calibrations
global birds

# Lists to accumulate info for different birds - Make these globals at top of app?
global birds_datetime_starts
global birds_datetime_ends
global birds_data_means
global birds_cal_means
global birds_details

### now make them
birds_datetime_starts = []
birds_datetime_ends = []
birds_data_means = []
birds_cal_means = []
birds_details = []
 
global cal_gradient
global cal_intercept
global baseline_cal_mean
global cal_r_squared

## globals from exec opne
global cal1_value
global cal2_value
global cal3_value

# Default figure view
global default_figure_width
global default_figure_height


# this declares some variables that are essentially globals once declared
# NOTE you will need to edit the TEMPLATE_set_user_values.py file and
#   save it as a set_user_values.py file
exec(open("set_user_values.py").read())

################
# Function Set_Globals to declare global variables all in one place - is this possible?
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    About: run at startup, but not yet doing that, using definitions above only
#######
def Set_Globals():
    # general info about the fle
    global user_INPATH
    global user_BURROW
    global data_DATE

    global data ## this  is a dataframe to be defined later
    global calibrations

    global cal_gradient
    global cal_intercept
    global baseline_cal_mean
    global cal_r_squared

# A class defining an object that stores axes limits for
# pyplot displays
class AxesLimits():
    def __init__(self, xstart, xend, ystart, yend):
        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend

# A function to set a pair of draggle points on an interactive trace plot
# RETURNS
#       mean, markers, isGood
#       mean -- mean strain measurement value between two marked points
#       markers -- dataframe of marker information, including start and end index on the trace
#       isGood -- boolean confirmation that the plot wasn't closed before both points were marked
#       newAxesLimits -- bounding box limits for the plot view that was shown right before exiting 
def getTracePointPair(category, markers=None, axesLimits=None):

    # Print a message
    print("Add {category} start point, then press enter.".format(category=category))

    # Turn on the Matplotlib-Pyplot viewer
    # Shows the trace from the globally-defined data
    pyplot.ion()
    fig, ax = pyplot.subplots()  # added 111
    fig.set_size_inches((default_figure_width, default_figure_height)) 
    ax.plot(data.loc[:,"Measure"])

    #axcut = ax.axes([0.9, 0.0, 0.1, 0.075])
    #bcut = Button(axcut, 'YES', color='red', hovercolor='green')

    if (axesLimits is not None):
        ax.set_xlim(left=axesLimits.xstart, right=axesLimits.xend)
        ax.set_ylim(bottom=axesLimits.ystart, top=axesLimits.yend)

    if (markers is not None):
        # Add any previous markers
        annotateCurrentMarkers(markers)

    # Initialize the draggable markers
    dm = DraggableMarker(category=category, startY=min(data["Measure"]))

    pyplot.show(block=True)

    pyplot.ioff()

    # Gather the marked points data
    index_start = min(dm.index_start, dm.index_end)
    index_end = max(dm.index_start, dm.index_end)
    time_start = data.loc[index_start,"Datetime"]
    time_end = data.loc[index_end,"Datetime"]
    measures = data.loc[index_start:index_end,"Measure"]
    mean = statistics.mean(measures)

    # Extract the axes limits for the final interactive plot view
    # in case the user wants to use those limits to restore the view on the next plot
    endView_xstart, endView_xend = ax.get_xlim()
    endView_ystart, endView_yend = ax.get_ylim()
    newAxesLimits = AxesLimits(endView_xstart, endView_xend, endView_ystart, endView_yend)

    # Confirm the plot was not exited before both points were marked
    isGood = dm.isGood

    print("""
    Measured {category} from {start} to {end}.
    Mean {category} measurement is {mean}.
    """.format(category=category, start=time_start, end=time_end, mean=round(mean,2)))

    # Create a dataframe with information about the marked points
    markers = pandas.DataFrame({"Category":category,
                                "Point":["Start", "End"],
                                "Index":[index_start, index_end],
                                "Datetime":[time_start, time_end],
                                "Measure":[data.loc[index_start,"Measure"], data.loc[index_end,"Measure"]]})
    markers = markers.set_index("Index")

    return mean, markers, isGood, newAxesLimits

# A function to plot all markers from a markers dataframe on the current pyplot viewer
#   (to be used for the markers dataframe as returned by getTracePointPair)
def annotateCurrentMarkers(markers):
    ax = pyplot.gca()

    # Plot the pairs of marker points separately, so lines aren't drawn betwen them
    for l, df in markers.groupby("Category"):
        ax.plot(df.loc[:,"Measure"], marker="o", color="black", ms=8)
        for index, row in df.iterrows():
            label = "{category} {point}".format(category=df.loc[index,"Category"], point=df.loc[index, "Point"])
            ax.annotate(label, (index, df.loc[index, "Measure"]), rotation=60)

# A class for a set of draggable markers on a Matplotlib-pyplot line plot
#   Designed to record data from two separate markers, which the user confirms with an "enter key"
# Adapted from https://stackoverflow.com/questions/43982250/draggable-markers-in-matplotlib
class DraggableMarker():
    def __init__(self, category, startY, startX=0):
        self.isGood = False
        self.category = category

        self.index_start = 0
        self.index_end = 0

        self.buttonClassIndex = 0
        self.buttonClasses = ["{category} start".format(category=category), "{category} end".format(category=category)]

        self.ax = pyplot.gca()
        self.lines=self.ax.lines
        self.lines=self.lines[:]

        self.tx = [self.ax.text(0,0,"") for l in self.lines]
        self.marker = [self.ax.plot([startX],[startY], marker="o", color="red")[0]]

        self.draggable = False

        self.isZooming = False
        self.isPanning = False

        self.currX = 0
        self.currY = 0

        self.c0 = self.ax.figure.canvas.mpl_connect("key_press_event", self.key)
        self.c1 = self.ax.figure.canvas.mpl_connect("button_press_event", self.click)
        self.c2 = self.ax.figure.canvas.mpl_connect("button_release_event", self.release)
        self.c3 = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.drag)

    def click(self,event):
        if event.button==1 and not self.isPanning and not self.isZooming:
            #leftclick
            self.draggable=True
            self.update(event)
            [tx.set_visible(self.draggable) for tx in self.tx]
            [m.set_visible(self.draggable) for m in self.marker]
            self.ax.figure.canvas.draw_idle()        
                
    def drag(self, event):
        if self.draggable:
            self.update(event)
            self.ax.figure.canvas.draw_idle()

    def release(self,event):
        self.draggable=False
        
    def update(self, event):
        try:        
            line = self.lines[0]
            x,y = self.get_closest(line, event.xdata) 
            self.tx[0].set_position((x,y))
            self.tx[0].set_text(self.buttonClasses[self.buttonClassIndex])
            self.marker[0].set_data([x],[y])
            self.currX = x
            self.currY = y
        except TypeError:
            pass

    def get_closest(self,line, mx):
        x,y = line.get_data()
        try: 
            mini = np.argmin(np.abs(x-mx))
            return x[mini], y[mini]
        except TypeError:
            pass

    def key(self,event):
        if (event.key == 'o'):
            self.isZooming = not self.isZooming
            self.isPanning = False
        elif(event.key == 'p'):
            self.isPanning = not self.isPanning
            self.isZooming = False
        elif(event.key == 't'):
            # A custom re-zoom, now that 'r' goes to 
            # the opening view (which might be retained from a previous view)
            line = self.lines[0]
            full_xstart = min(line.get_xdata())
            full_xend = max(line.get_xdata())
            full_ystart = min(line.get_ydata())
            full_yend = max(line.get_ydata())
            self.ax.axis(xmin=full_xstart, xmax=full_xend, ymin=full_ystart, ymax=full_yend)
        elif (event.key == 'enter'):
            if(self.buttonClassIndex==0):
                self.ax.plot([self.currX],[self.currY], marker="o", color="yellow")
                self.buttonClassIndex=1
                self.index_start = self.currX
                print("Add {category} end point, then press enter.".format(category=self.category))
            elif(self.buttonClassIndex==1):
                self.index_end = self.currX
                self.isGood = True
                pyplot.close()
            self.update(event)



################
# Function Set_Startup 
#    RAM 7/26/22
#    parameters:NONE
#    returns: NONE
#    about: Housekeeping and welcome screen
#######
def Set_Startup():
    # general info about the app
    ### show welcome
    print("""Welcome to the Mass-O-Matic-O-Matic (M.O.M.O.M), a script for easy data entry of M.O.M. data.\n\n""")
    ## could also call Set_Globals here if that worked. not sure it does, so I don't yet

    # this declares some variables that are essentially globals once declared
    # NOTE you will need to edit the TEMPLATE_set_user_values.py file and
    #   save it as a set_user_values.py file
    exec(open("set_user_values.py").read())
    


################
# Function Get_File_Path to return path to file
#    RAM 7/25/22
#    Ask user for input name - eventually replace with file dialog
#    returns the system-specific path to the file of interest
#       other - should have Set_Globals before using this
#######
def Get_File_Info():
    # just get the path for the file and other user info
    # return the path and burrow
    #   BURROW not being defined
    global user_INPATH
    global user_BURROW
    
    if (True):
        my_user_INPATH = "north_end_7_20.TXT"
    else:
        my_user_INPATH = input("**Enter input file:    ")

    user_BURROW = str(input("**Enter burrow number: "))
    print(my_user_INPATH)
    print(user_BURROW)
    
    return my_user_INPATH
    
    
################
# Function Load_data to use file path to find and load, then return dataframe 
#    RAM 7/25/22
#    parameters: my_Path is the 
#    returns the prepped dataframe and the user path
#######
def Load_data():
    global data_DATE
    global user_INPATH

    try:
        user_INPATH = Get_File_Info()
    
        my_data = pandas.read_csv(user_INPATH, header=None, names=["Measure", "Datetime"], 
                            encoding="utf-8", encoding_errors="replace", on_bad_lines="skip", 
                            engine="python")
        my_data["Measure"] = pandas.to_numeric(my_data["Measure"], errors="coerce")
        my_data["Datetime"] = pandas.to_datetime(my_data["Datetime"], utc=True, errors="coerce")

        # We've possibly forced parsing of some malformed data
        #   ("replace" utf-8 encoding errors in read_csv() 
        #     and "coerce" datetime errors in to_numeric() and to_datetime()), so 
        #   now we need to clean that up.
        # Simply drop all points from the file where Measure has been coerced to NaN
        #   and where Datetime has been coerced to NaT
        my_data = my_data[~my_data.Measure.isnull() & ~my_data.Datetime.isnull()]

    except FileNotFoundError:
        sys.exit("Input file not found. Exiting. FileNotFoundError")
    except pandas.errors.EmptyDataError:
        sys.exit("Error parsing empty file. Exiting. EmptyDataError")
    except pandas.errors.ParserError:
        sys.exit("Error parsing input file. Exiting. ParserError")
    except Exception as e:
        sys.exit("Error parsing input file. Exiting. {}".format(e))

        # Display input information
        # data_DATE not being defined correctly
    data_DATE = my_data.Datetime.iloc[-1].date()
    # print("Working with data ending on: " # {date} in burrow.".format(date=data_DATE))
    # print(data_DATE)

    return my_data, user_INPATH  #, user_BURROW  # , burrow_number

################
# Function my_Do_Calibrations get the calibration for the MOM on this burrow-night
#    RAM 7/25/22
#    parameters: my_dataframe -> data to work with
#       - 
#    returns tuple with information about the result of the calculations
#######
def my_Do_Calibrations(my_dataframe):
    ### assign passed dataframe to the global data
    global data
    data = my_dataframe
    
    global calibrations

    global cal_gradient
    global cal_intercept
    global baseline_cal_mean
    global cal_r_squared

    #
    # CALIBRATIONS
    #
    print("""
    Displaying M.O.M. data from {ipath}.
    Press 'r' to reset view to starting view.
    Press 't' to reset view to data limits.
    Press 'o' to rectangle zoom.
    Press 'p' to pan.
    Press 'q' to quit program.
    """.format(ipath=user_INPATH))

    # Add baselines
    baseline_cal_mean, baseline_cal_markers, baseline_cal_Good, axesLimits = getTracePointPair("Baseline")
    markers = baseline_cal_markers

    # Add calibrations as separate pairs of points
    cal1_mean, cal1_markers, cal1_Good, axesLimits = getTracePointPair("Cal1[{}]".format(cal1_value), markers, axesLimits)
    markers = pandas.concat([markers, cal1_markers])

    cal2_mean, cal2_markers, cal2_Good, axesLimits = getTracePointPair("Cal2[{}]".format(cal2_value), markers, axesLimits)
    markers = pandas.concat([markers, cal2_markers])

    cal3_mean, cal3_markers, cal3_Good, axesLimits = getTracePointPair("Cal3[{}]".format(cal3_value), markers, axesLimits)
    markers = pandas.concat([markers, cal3_markers])

    # Check all the calibrations were marked successfully
    if (not baseline_cal_Good or not cal1_Good or not cal2_Good or not cal3_Good):
        sys.exit("""Error! Calibration points not set well. Try again. 
                    (Did you exit out of a calibration window?)""")

    # Clean up the marked calibration points data
    calibrations = pandas.DataFrame({"Category":["Cal1", "Cal2", "Cal3"],
                                    "Value_True":[cal1_value, cal2_value, cal3_value],
                                    "Value_Measured":[cal1_mean, cal2_mean, cal3_mean]})
    calibrations["Value_Difference"] = abs(calibrations["Value_Measured"] - baseline_cal_mean)

    # Get the linear regression information across the three calibration points
    cal_gradient, cal_intercept, cal_r_value, cal_p_value, cal_std_err = stats.linregress(calibrations["Value_Difference"], calibrations["Value_True"])
    cal_r_squared = cal_r_value**2

    # A tiny function to confirm if we want to continue
    #   after showing the calibration plot results. Used just below.
    def continueKey(event):
        if(event.key == "y"):
            pyplot.close()
        if(event.key == "n"):
            sys.exit("Cancelled manually after calibration. No results recorded.")

    print("Showing calibration results.\nPress 'y' to proceed or 'n' to exit.")
    fig, ax = pyplot.subplots()
    fig.canvas.mpl_connect('key_press_event', continueKey)
    ax.plot(calibrations["Value_Difference"], calibrations["Value_True"], marker="o", color="black", linestyle="None")
    ax.plot(calibrations["Value_Difference"], calibrations["Value_Difference"]*cal_gradient+cal_intercept, color="gray", linestyle="dashed")
    pyplot.xlabel("Measured value (strain difference from baseline)")
    pyplot.ylabel("True value (g)")
    pyplot.title("Calibration regression\n(R^2={r}, Inter={i}, Slope={s})".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5)))
    pyplot.show()

################
# Function Do_Birds get the calibration and bird weight data for a single bird in a MOM file
#    RAM 7/25/22
#    parameters: NONE
#    returns: NONE
#######
def Do_Bird():
    
        # If the user wants to a bird, first get the calibration points for that bird
        bird_cal_mean, bird_cal_markers, bird_cal_good, bird_cal_axesLimits = getTracePointPair("Calibration[Bird]")

        bird_data_mean, bird_data_markers, bird_data_good, bird_data_axesLimits = getTracePointPair("Bird Data", bird_cal_markers, bird_cal_axesLimits)
        measure_start = bird_data_markers[bird_data_markers["Point"]=="Start"].Datetime.iloc[0]
        measure_end = bird_data_markers[bird_data_markers["Point"]=="End"].Datetime.iloc[0]

        # Allow the user to input extra details for a "Notes" column
        bird_details = input("Enter any details about the bird:      ")

        # Add the info about this bird to the accumulating lists
        birds_datetime_starts.append(measure_start)
        birds_datetime_ends.append(measure_end)
        birds_data_means.append(bird_data_mean)
        birds_cal_means.append(bird_cal_mean)
        birds_details.append(bird_details)

################
# Function Set_Globals to declare global variables all in one place - is this possible?
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    About: run at startup, but not yet doing that, using definitions above only
#######
def Do_Multiple_Birds():
    global birds
    # assumes have lists declared as global
    # Allow the user to continue entering birds
    while (True):
        response = input("Would you like to enter data for a particular bird? (y/n)    ")
        if (response == "y"):
            # If the user wants to a bird, first get the calibration points for that bird
            Do_Bird()
        elif(response == "n"):
            # Carry on if the user doesn't want more info
            break

    # Make the accumulated bird info into a clean dataframe for exporting
    # NOTE: Why don't I have user_BURROW and data_Date defined? it was a global, but throw an error here, so I have commented it out and replace wtih dummy
    birds = pandas.DataFrame({"Burrow":user_BURROW,
                            "Date":data_DATE,
                            "Datetime_Measure_Start":birds_datetime_starts,
                            "Datetime_Measure_End":birds_datetime_ends,
                            "Mean_Data_Strain":birds_data_means,
                            "Mean_Calibration_Strain":birds_cal_means,
                            "Details":birds_details})

    # Convert the Datetime columns back to character strings for exporting
    birds["Datetime_Measure_Start"] = birds["Datetime_Measure_Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    birds["Datetime_Measure_End"] = birds["Datetime_Measure_End"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Calculate the baseline and regressed mass estimates for the birds
    birds["Baseline_Difference"] = abs(birds["Mean_Data_Strain"] - birds["Mean_Calibration_Strain"]) 
    birds["Regression_Mass"] = birds["Baseline_Difference"] * cal_gradient + cal_intercept
    # ADDED for yucks
    print("Bird calculated masses: ")
    print(birds["Regression_Mass"])
    ### END ADDED for Yucks

    print("Bird entries complete.")

################
# Function Output_MOM_Data to declare global variables all in one place - is this possible?
#    RAM 7/26/22
#    Parameters: NONE
#    Returns: NONE
#    About: sends accumulated data to a csv file
#######
def Output_MOM_Data():
        # Ask the user for any last summary details for the summary info sheet
    summaryDetails = input("Enter any summary details:\n")

    # Export summary info, including calibration info, to file
    path_summary = "Burrow_{burrow}_{date}_SUMMARY.txt".format(burrow=user_BURROW, date=data_DATE)
    with open(path_summary, 'w') as f:
        f.write("M.O.M. Results\n")
        f.write("Burrow: {burrow}\n".format(burrow=user_BURROW))
        f.write("Deployment date: {date}\n".format(date=data_DATE))
        f.write("Number of birds recorded: {numBirds}\n".format(numBirds=len(birds_data_means)))
        f.write("\n\n\nCalibration details:\n")
        f.write("Mean value from baseline for calibration: {}\n\n".format(baseline_cal_mean))
        f.write(calibrations.to_csv(sep="\t", index=False))
        f.write("\nCalibration regression\nR^2={r}, Intercept={i}, Slope={s}".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5)))
        f.write("\n\n\n")
        f.write("Summary details:\n")
        f.write(summaryDetails)
        f.close()

    print("Wrote summary details, including calibration info, to\n\t\t{spath}".format(spath=path_summary))

    # Export bird info (if any was added)
    path_bird = "Burrow_{burrow}_{date}_BIRDS.txt".format(burrow=user_BURROW, date=data_DATE)

    if (len(birds_data_means) > 0):
        birds.to_csv(path_bird, index=False)
        print("Wrote bird details to\n\t\t{bpath}".format(bpath=path_bird))
    else:
        print("No birds recorded.")

#################################        
# START the app
########

# Get started...
Set_Startup()  

# Load dataset of interest
my_data, user_INPATH = Load_data()


# CALIBRATIONS
my_Do_Calibrations(my_data)

# BIRD ENTRY
Do_Multiple_Birds()

# Send data to files
Output_MOM_Data()


#
# END PROGRAM - don't really need this eventually
#
print("Exiting")
sys.exit()
