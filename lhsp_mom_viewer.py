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
import numpy as np
import statistics
from scipy import stats 


exec(open("set_calibration_values.py").read())

# A function to set a pair of draggle points on an interactive trace plot
# RETURNS
#       mean, markers, isGood
#       mean -- mean strain measurement value between two marked points
#       markers -- dataframe of marker information, including start and end index on the trace
#       isGood -- boolean confirmation that the plot wasn't closed before both points were marked
def getTracePointPair(category, markers=None):

    # Print a message
    print("Add {category} start point, then press enter.".format(category=category))

    # Turn on the Matplotlib-Pyplot viewer
    # Shows the trace from the globally-defined data
    pyplot.ion()
    fig, ax = pyplot.subplots()
    ax.plot(data.loc[:,"Measure"])

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

    return mean, markers, isGood

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
            for i, line in enumerate(self.lines):
                # foolish way to do this, but keeps it flexible in case
                # the code needs to be adapted for multiple lines
                if (i>0):
                    continue
                x,y = self.get_closest(line, event.xdata) 
                self.tx[i].set_position((x,y))
                self.tx[i].set_text(self.buttonClasses[self.buttonClassIndex])
                self.marker[i].set_data([x],[y])
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
#        
# START RUNNING SCRIPT
#
print("""
Welcome to the Mass-O-Matic-O-Matic (M.O.M.O.M), 
a script for easy data entry of M.O.M. data.\n\n""")

#
# USER SETUP
#
user_INPATH = input("**Enter input file:    ")
try:
    data = pandas.read_csv(user_INPATH, header=None, names=["Measure", "Datetime"])
    data["Datetime"] = pandas.to_datetime(data["Datetime"])
except FileNotFoundError:
    sys.exit("Input file not found. Exiting.")
except pandas.errors.EmptyDataError:
    sys.exit("Error parsing empty file. Exiting.")
except pandas.errors.ParserError:
    sys.exit("Error parsing input file. Exiting.")

print("Read in {input}.\n".format(input=user_INPATH))

data_DATE = data.Datetime.iloc[-1].date()

user_BURROW = input("**Enter burrow number: ")

# Display input information
print("Working with data ending on {date} in burrow {burrow}.".format(date=data_DATE, burrow=user_BURROW))

#
# CALIBRATIONS
#
print("""
Displaying M.O.M. data from {ipath}.
Press 'r' to reset view.
Press 'o' to rectangle zoom.
Press 'p' to pan.
Press 'q' to quit program.
""".format(ipath=user_INPATH))

# Add baselines
baseline_mean, baseline_markers, baselineGood = getTracePointPair("Baseline")
markers = baseline_markers

# Add calibrations as separate pairs of points
cal1_mean, cal1_markers, cal1Good = getTracePointPair("Cal1[{}]".format(cal1_value), markers)
markers = pandas.concat([markers, cal1_markers])

cal2_mean, cal2_markers, cal2Good = getTracePointPair("Cal2[{}]".format(cal2_value), markers)
markers = pandas.concat([markers, cal2_markers])

cal3_mean, cal3_markers, cal3Good = getTracePointPair("Cal3[{}]".format(cal3_value), markers)
markers = pandas.concat([markers, cal3_markers])

# Check all the calibrations were marked successfully
if (not baselineGood or not cal1Good or not cal2Good or not cal3Good):
    sys.exit("""Error! Calibration points not set well. Try again. 
                (Did you exit out of a calibration window?)""")

# Clean up the marked calibration points data
calibrations = pandas.DataFrame({"Category":["Cal1", "Cal2", "Cal3"],
                                 "Value_True":[cal1_value, cal2_value, cal3_value],
                                 "Value_Measured":[cal1_mean, cal2_mean, cal3_mean]})
calibrations["Value_Difference"] = abs(calibrations["Value_Measured"] - baseline_mean)

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

#
# BIRD ENTRY
#

# Lists to accumulate info for different birds
birds_datetime_starts = []
birds_datetime_ends = []
birds_means = []
birds_details = []

# Allow the user to continue entering birds
while (True):
    response = input("Would you like to enter information for a bird? (y/n)    ")
    if (response == "y"):

        # If the user wants to enter another bird, get the pair of points spanning the measure
        bird_mean, bird_markers, bird_good = getTracePointPair("Bird")
        measure_start = bird_markers[bird_markers["Point"]=="Start"].Datetime.iloc[0]
        measure_end = bird_markers[bird_markers["Point"]=="End"].Datetime.iloc[0]

        # Allow the user to input extra details for a "Notes" column
        details = input("Enter any details about the bird:      ")

        # Add the info about this bird to the accumulating lists
        birds_datetime_starts.append(measure_start)
        birds_datetime_ends.append(measure_end)
        birds_means.append(bird_mean)
    elif(response == "n"):
        # Carry on if the user doesn't want more info
        break

# Make the accumulated bird info into a clean dataframe for exporting
birds = pandas.DataFrame({"Burrow":user_BURROW,
                          "Date":data_DATE,
                          "Datetime_Measure_Start":birds_datetime_starts,
                          "Datetime_Measure_End":birds_datetime_ends,
                          "Mean_Strain_Value":birds_means,
                          "Details":details})

# Convert the Datetime columns back to character strings for exporting
birds["Datetime_Measure_Start"] = birds["Datetime_Measure_Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
birds["Datetime_Measure_End"] = birds["Datetime_Measure_End"].dt.strftime("%Y-%m-%d %H:%M:%S")

# Calculate the baseline and regressed mass estimates for the birds
birds["Baseline_Difference"] = abs(birds["Mean_Strain_Value"] - baseline_mean) 
birds["Regression_Mass"] = birds["Baseline_Difference"] * cal_gradient + cal_intercept
print("Bird entries complete.")

#
# OUTPUT
#

# Ask the user for any last summary details for the summary info sheet
summaryDetails = input("Enter any summary details:\n")

# Export summary info, including calibration info, to file
path_summary = "Burrow_{burrow}_{date}_SUMMARY.txt".format(burrow=user_BURROW, date=data_DATE)
with open(path_summary, 'w') as f:
    f.write("M.O.M. Results\n")
    f.write("Burrow: {burrow}\n".format(burrow=user_BURROW))
    f.write("Deployment date: {date}\n".format(date=data_DATE))
    f.write("Number of birds recorded: {numBirds}\n".format(numBirds=len(birds_means)))
    f.write("\n\n\nCalibration details:\n")
    f.write(calibrations.to_csv(sep="\t", index=False))
    f.write("\nCalibration regression\nR^2={r}, Intercept={i}, Slope={s})".format(r=round(cal_r_squared,5), i=round(cal_intercept,5), s=round(cal_gradient,5)))
    f.write("\n\n\n")
    f.write("Summary details:\n")
    f.write(summaryDetails)
    f.close()

print("Wrote summary details, including calibration info, to\n\t\t{spath}".format(spath=path_summary))

# Export bird info (if any was added)
path_bird = "Burrow_{burrow}_{date}_BIRDS.txt".format(burrow=user_BURROW, date=data_DATE)

if (len(birds_means) > 0):
    birds.to_csv(path_bird, index=False)
    print("Wrote bird details to\n\t\t{bpath}".format(bpath=path_bird))
else:
    print("No birds recorded.")

#
# END PROGRAM
#
print("Exiting")
sys.exit()