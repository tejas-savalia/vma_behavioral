#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2020.2.4),
    on October 14, 2020, at 09:21
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2020.2.4'
expName = 'vm_20xx'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sort_keys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\Tejas\\Documents\\Lab-Codes\\Visuomotor_Adaptation_psychopy_tablet\\no_feedback\\vm_20xx.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowGUI=False, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "instructions"
instructionsClock = core.Clock()
Welcome = visual.TextStim(win=win, name='Welcome',
    text='Welcome to Visuomotor Adaptation!\n\nYour task is to move a circle controlled by the pen to the target cross.  \n\nThe circle will disappear once you start to move and reappear as you cover the distance required to hit the target. \n\nTry to hit the target as many times as possible. \n\nScratch on the tab using the pen to proceed to a demo.',
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
mouse = event.Mouse(win=win)
x, y = [None, None]
mouse.mouseClock = core.Clock()

# Initialize components for Routine "break_0"
break_0Clock = core.Clock()
text = visual.TextStim(win=win, name='text',
    text="Let's try four trials.\n\nScratch the tablet to continue.",
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
mouse_2 = event.Mouse(win=win)
x, y = [None, None]
mouse_2.mouseClock = core.Clock()

# Initialize components for Routine "demo"
demoClock = core.Clock()
def euclidean_dist(vec1, vec2):
    return sqrt((vec1[0] - vec2[0])**2 + (vec1[1] - vec2[1])**2)
    

demo_enclosing_1 = visual.Polygon(
    win=win, name='demo_enclosing_1',
    edges=64, size=(0.9, 0.9),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)
demo_enclosing = visual.Polygon(
    win=win, name='demo_enclosing',
    edges=64, size=(0.80, 0.80),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[0,0,0], fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)
demo_target = visual.ShapeStim(
    win=win, name='demo_target', vertices='cross',
    size=(0.05, 0.05),
    ori=45, pos=[0,0],
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
demo_fixation = visual.Polygon(
    win=win, name='demo_fixation',
    edges=32, size=(0.025, 0.025),
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,0,0], lineColorSpace='rgb',
    fillColor=[1,0,0], fillColorSpace='rgb',
    opacity=1, depth=-4.0, interpolate=True)
demo_mouse = event.Mouse(win=win)
x, y = [None, None]
demo_mouse.mouseClock = core.Clock()

# Initialize components for Routine "demo_feedback"
demo_feedbackClock = core.Clock()
demo_enclosing_feedback_1 = visual.Polygon(
    win=win, name='demo_enclosing_feedback_1',
    edges=128, size=(0.9, 0.9),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[0, 0, 0], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)
demo_enclosing_feedback = visual.Polygon(
    win=win, name='demo_enclosing_feedback',
    edges=64, size=(0.8, 0.8),
    ori=0, pos=(0, 0),
    lineWidth=1, lineColor=[-1,-1,-1], lineColorSpace='rgb',
    fillColor=[0, 0, 0], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)
demo_target_feedback = visual.ShapeStim(
    win=win, name='demo_target_feedback', vertices='cross',
    size=(0.05, 0.05),
    ori=45, pos=[0,0],
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=-2.0, interpolate=True)
demo_fixation_feedback = visual.Polygon(
    win=win, name='demo_fixation_feedback',
    edges=32, size=(0.025, 0.025),
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,0,0], lineColorSpace='rgb',
    fillColor=[1,0,0], fillColorSpace='rgb',
    opacity=1, depth=-3.0, interpolate=True)
text_4 = visual.TextStim(win=win, name='text_4',
    text='default text',
    font='Arial',
    pos=(0.4, 0.4), height=0.02, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=-5.0);

# Initialize components for Routine "experiment_start"
experiment_startClock = core.Clock()
text_3 = visual.TextStim(win=win, name='text_3',
    text="Experiment starts now!\n\nTry to come as close to the target as possible even if you can't hit it.\n\nScratch the trackpad to continue.",
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
mouse_6 = event.Mouse(win=win)
x, y = [None, None]
mouse_6.mouseClock = core.Clock()

# Initialize components for Routine "baseline"
baselineClock = core.Clock()
baseline_target = visual.ShapeStim(
    win=win, name='baseline_target', vertices='cross',
    size=(0.05, 0.05),
    ori=45, pos=[0,0],
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)
baseline_fixation = visual.Polygon(
    win=win, name='baseline_fixation',
    edges=32, size=(0.025, 0.025),
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,0,0], lineColorSpace='rgb',
    fillColor=[1,0,0], fillColorSpace='rgb',
    opacity=1.0, depth=-1.0, interpolate=True)
baseline_mouse = event.Mouse(win=win)
x, y = [None, None]
baseline_mouse.mouseClock = core.Clock()

# Initialize components for Routine "baseline_feedback"
baseline_feedbackClock = core.Clock()
baseline_target_feedback = visual.ShapeStim(
    win=win, name='baseline_target_feedback', vertices='cross',
    size=(0.05, 0.05),
    ori=45, pos=[0,0],
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)
baseline_fixation_feedback = visual.Polygon(
    win=win, name='baseline_fixation_feedback',
    edges=32, size=(0.025, 0.025),
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,0,0], lineColorSpace='rgb',
    fillColor=[1,0,0], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)

# Initialize components for Routine "break_1"
break_1Clock = core.Clock()
text_2 = visual.TextStim(win=win, name='text_2',
    text="Start next block.\n\nRemember to try to be as accurate as possible by getting close to the target as many times as possible.\n\nIf it helps, take your time to make the movement and be accurate.\n\nScratch the tablet to continue whenever you're ready.\n\n ",
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
mouse_3 = event.Mouse(win=win)
x, y = [None, None]
mouse_3.mouseClock = core.Clock()

# Initialize components for Routine "rotated"
rotatedClock = core.Clock()
rotated_target = visual.ShapeStim(
    win=win, name='rotated_target', vertices='cross',
    size=(0.05, 0.05),
    ori=45, pos=[0,0],
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)
rotated_fixation = visual.Polygon(
    win=win, name='rotated_fixation',
    edges=32, size=(0.025, 0.025),
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,0,0], lineColorSpace='rgb',
    fillColor=[1,0,0], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)
rotated_mouse = event.Mouse(win=win)
x, y = [None, None]
rotated_mouse.mouseClock = core.Clock()
def rotate(x, y, angle):
    radians = (pi/180)*angle
    cos_val = cos(radians)
    sin_val = sin(radians)
    nx = x*cos_val + y*sin_val
    ny = y*cos_val - x*sin_val
    return [nx, ny];
    
count = 0

# Initialize components for Routine "rotated_feedback"
rotated_feedbackClock = core.Clock()
rotated_target_feedback = visual.ShapeStim(
    win=win, name='rotated_target_feedback', vertices='cross',
    size=(0.05, 0.05),
    ori=45, pos=[0,0],
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)
rotated_fixation_feedback = visual.Polygon(
    win=win, name='rotated_fixation_feedback',
    edges=32, size=(0.025, 0.025),
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,0,0], lineColorSpace='rgb',
    fillColor=[1,0,0], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)

# Initialize components for Routine "break_2"
break_2Clock = core.Clock()
rotated_breaks = visual.TextStim(win=win, name='rotated_breaks',
    text="Take a break.\n\nRemember to try to be as accurate as possible by getting as close to the target coss as possible.\n\nIf it helps, take your time to make the movement and be accurate.\n\nScratch the tablet to continue whenever you're ready.\n\n ",
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
mouse_5 = event.Mouse(win=win)
x, y = [None, None]
mouse_5.mouseClock = core.Clock()

# Initialize components for Routine "transfer"
transferClock = core.Clock()
transfer_target = visual.ShapeStim(
    win=win, name='transfer_target', vertices='cross',
    size=(0.05, 0.05),
    ori=45, pos=[0,0],
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)
transfer_fixation = visual.Polygon(
    win=win, name='transfer_fixation',
    edges=32, size=(0.025, 0.025),
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,0,0], lineColorSpace='rgb',
    fillColor=[1,0,0], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)
transfer_mouse = event.Mouse(win=win)
x, y = [None, None]
transfer_mouse.mouseClock = core.Clock()

# Initialize components for Routine "transfer_feedback"
transfer_feedbackClock = core.Clock()
transfer_target_feedback = visual.ShapeStim(
    win=win, name='transfer_target_feedback', vertices='cross',
    size=(0.05, 0.05),
    ori=45, pos=[0,0],
    lineWidth=1, lineColor=[1,1,1], lineColorSpace='rgb',
    fillColor=[1,1,1], fillColorSpace='rgb',
    opacity=1, depth=0.0, interpolate=True)
transfer_fixation_feedback = visual.Polygon(
    win=win, name='transfer_fixation_feedback',
    edges=32, size=(0.025, 0.025),
    ori=0, pos=[0,0],
    lineWidth=1, lineColor=[1,0,0], lineColorSpace='rgb',
    fillColor=[1,0,0], fillColorSpace='rgb',
    opacity=1, depth=-1.0, interpolate=True)

# Initialize components for Routine "Done"
DoneClock = core.Clock()
done_text = visual.TextStim(win=win, name='done_text',
    text='You are all set!\n\nThank you for participating.\n\nPlease see the researcher.',
    font='Arial',
    pos=(0, 0), height=0.04, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1, 
    languageStyle='LTR',
    depth=0.0);
mouse_4 = event.Mouse(win=win)
x, y = [None, None]
mouse_4.mouseClock = core.Clock()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# ------Prepare to start Routine "instructions"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse
gotValidClick = False  # until a click is received
# keep track of which components have finished
instructionsComponents = [Welcome, mouse]
for thisComponent in instructionsComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
instructionsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "instructions"-------
while continueRoutine:
    # get current time
    t = instructionsClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=instructionsClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Welcome* updates
    if Welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        Welcome.frameNStart = frameN  # exact frame index
        Welcome.tStart = t  # local t and not account for scr refresh
        Welcome.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(Welcome, 'tStartRefresh')  # time at next scr refresh
        Welcome.setAutoDraw(True)
    # *mouse* updates
    if mouse.status == NOT_STARTED and t >= 0-frameTolerance:
        # keep track of start time/frame for later
        mouse.frameNStart = frameN  # exact frame index
        mouse.tStart = t  # local t and not account for scr refresh
        mouse.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
        mouse.status = STARTED
        mouse.mouseClock.reset()
        prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
    if mouse.status == STARTED:  # only update if started and not finished!
        buttons = mouse.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                continueRoutine = False    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "instructions"-------
for thisComponent in instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('Welcome.started', Welcome.tStartRefresh)
thisExp.addData('Welcome.stopped', Welcome.tStopRefresh)
# store data for thisExp (ExperimentHandler)
thisExp.addData('mouse.started', mouse.tStart)
thisExp.addData('mouse.stopped', mouse.tStop)
thisExp.nextEntry()
# the Routine "instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
trials = data.TrialHandler(nReps=2, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='trials')
thisExp.addLoop(trials)  # add the loop to the experiment
thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
if thisTrial != None:
    for paramName in thisTrial:
        exec('{} = thisTrial[paramName]'.format(paramName))

for thisTrial in trials:
    currentLoop = trials
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "break_0"-------
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_2
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    break_0Components = [text, mouse_2]
    for thisComponent in break_0Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    break_0Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "break_0"-------
    while continueRoutine:
        # get current time
        t = break_0Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=break_0Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            text.setAutoDraw(True)
        # *mouse_2* updates
        if mouse_2.status == NOT_STARTED and t >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            mouse_2.frameNStart = frameN  # exact frame index
            mouse_2.tStart = t  # local t and not account for scr refresh
            mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
            mouse_2.status = STARTED
            mouse_2.mouseClock.reset()
            prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
        if mouse_2.status == STARTED:  # only update if started and not finished!
            buttons = mouse_2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    continueRoutine = False        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_0Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "break_0"-------
    for thisComponent in break_0Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    trials.addData('text.started', text.tStartRefresh)
    trials.addData('text.stopped', text.tStopRefresh)
    # store data for trials (TrialHandler)
    trials.addData('mouse_2.started', mouse_2.tStart)
    trials.addData('mouse_2.stopped', mouse_2.tStop)
    # the Routine "break_0" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_2 = data.TrialHandler(nReps=1, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions_demo.xlsx'),
        seed=None, name='trials_2')
    thisExp.addLoop(trials_2)  # add the loop to the experiment
    thisTrial_2 = trials_2.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
    if thisTrial_2 != None:
        for paramName in thisTrial_2:
            exec('{} = thisTrial_2[paramName]'.format(paramName))
    
    for thisTrial_2 in trials_2:
        currentLoop = trials_2
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_2.rgb)
        if thisTrial_2 != None:
            for paramName in thisTrial_2:
                exec('{} = thisTrial_2[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "demo"-------
        continueRoutine = True
        # update component parameters for each repeat
        demo_fixation.opacity = 1
        demo_fixation.pos = [0, 0]
        first = True
        jitter = np.random.uniform(0.5, 1.5)
        demo_target.setPos((target_x, target_y))
        demo_fixation.setPos((0, 0))
        # setup some python lists for storing info about the demo_mouse
        demo_mouse.x = []
        demo_mouse.y = []
        demo_mouse.leftButton = []
        demo_mouse.midButton = []
        demo_mouse.rightButton = []
        demo_mouse.time = []
        gotValidClick = False  # until a click is received
        # keep track of which components have finished
        demoComponents = [demo_enclosing_1, demo_enclosing, demo_target, demo_fixation, demo_mouse]
        for thisComponent in demoComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        demoClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "demo"-------
        while continueRoutine:
            # get current time
            t = demoClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=demoClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            win.mouseVisible = False
            if demo_mouse.getPressed()[0]:
                if first:
                    mouse_center = demo_mouse.getPos()
                    first = False
                demo_fixation.opacity = 0
                if euclidean_dist(demo_mouse.getPos() - mouse_center, [0, 0]) > 0.41:
                    demo_fixation.pos = demo_mouse.getPos() - mouse_center
                    demo_fixation.opacity = 1
                    continueRoutine = False
            else:
                first = True
                demo_fixation.opacity = 1
                demo_fixation.pos = [0, 0]
                demo_mouse.setPos([0, 0])
            
            # *demo_enclosing_1* updates
            if demo_enclosing_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                demo_enclosing_1.frameNStart = frameN  # exact frame index
                demo_enclosing_1.tStart = t  # local t and not account for scr refresh
                demo_enclosing_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(demo_enclosing_1, 'tStartRefresh')  # time at next scr refresh
                demo_enclosing_1.setAutoDraw(True)
            
            # *demo_enclosing* updates
            if demo_enclosing.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                demo_enclosing.frameNStart = frameN  # exact frame index
                demo_enclosing.tStart = t  # local t and not account for scr refresh
                demo_enclosing.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(demo_enclosing, 'tStartRefresh')  # time at next scr refresh
                demo_enclosing.setAutoDraw(True)
            
            # *demo_target* updates
            if demo_target.status == NOT_STARTED and tThisFlip >= jitter-frameTolerance:
                # keep track of start time/frame for later
                demo_target.frameNStart = frameN  # exact frame index
                demo_target.tStart = t  # local t and not account for scr refresh
                demo_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(demo_target, 'tStartRefresh')  # time at next scr refresh
                demo_target.setAutoDraw(True)
            
            # *demo_fixation* updates
            if demo_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                demo_fixation.frameNStart = frameN  # exact frame index
                demo_fixation.tStart = t  # local t and not account for scr refresh
                demo_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(demo_fixation, 'tStartRefresh')  # time at next scr refresh
                demo_fixation.setAutoDraw(True)
            # *demo_mouse* updates
            if demo_mouse.status == NOT_STARTED and t >= jitter-frameTolerance:
                # keep track of start time/frame for later
                demo_mouse.frameNStart = frameN  # exact frame index
                demo_mouse.tStart = t  # local t and not account for scr refresh
                demo_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(demo_mouse, 'tStartRefresh')  # time at next scr refresh
                demo_mouse.status = STARTED
                demo_mouse.mouseClock.reset()
                prevButtonState = demo_mouse.getPressed()  # if button is down already this ISN'T a new click
            if demo_mouse.status == STARTED:  # only update if started and not finished!
                x, y = demo_mouse.getPos()
                demo_mouse.x.append(x)
                demo_mouse.y.append(y)
                buttons = demo_mouse.getPressed()
                demo_mouse.leftButton.append(buttons[0])
                demo_mouse.midButton.append(buttons[1])
                demo_mouse.rightButton.append(buttons[2])
                demo_mouse.time.append(demo_mouse.mouseClock.getTime())
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in demoComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "demo"-------
        for thisComponent in demoComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        demo_fixation.opacity = 1
        demo_fix_end_pos = demo_fixation.pos
        trials_2.addData('demo_enclosing_1.started', demo_enclosing_1.tStartRefresh)
        trials_2.addData('demo_enclosing_1.stopped', demo_enclosing_1.tStopRefresh)
        trials_2.addData('demo_enclosing.started', demo_enclosing.tStartRefresh)
        trials_2.addData('demo_enclosing.stopped', demo_enclosing.tStopRefresh)
        trials_2.addData('demo_target.started', demo_target.tStartRefresh)
        trials_2.addData('demo_target.stopped', demo_target.tStopRefresh)
        trials_2.addData('demo_fixation.started', demo_fixation.tStartRefresh)
        trials_2.addData('demo_fixation.stopped', demo_fixation.tStopRefresh)
        # store data for trials_2 (TrialHandler)
        trials_2.addData('demo_mouse.x', demo_mouse.x)
        trials_2.addData('demo_mouse.y', demo_mouse.y)
        trials_2.addData('demo_mouse.leftButton', demo_mouse.leftButton)
        trials_2.addData('demo_mouse.midButton', demo_mouse.midButton)
        trials_2.addData('demo_mouse.rightButton', demo_mouse.rightButton)
        trials_2.addData('demo_mouse.time', demo_mouse.time)
        trials_2.addData('demo_mouse.started', demo_mouse.tStart)
        trials_2.addData('demo_mouse.stopped', demo_mouse.tStop)
        # the Routine "demo" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "demo_feedback"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        demo_target_feedback.setPos((target_x, target_y))
        demo_fixation_feedback.setPos(demo_fix_end_pos)
        error = euclidean_dist(demo_fix_end_pos, demo_target_feedback.pos())
        score = error/0.84
        text_4.setText('Your score: ' + str(score))
        # keep track of which components have finished
        demo_feedbackComponents = [demo_enclosing_feedback_1, demo_enclosing_feedback, demo_target_feedback, demo_fixation_feedback, text_4]
        for thisComponent in demo_feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        demo_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "demo_feedback"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = demo_feedbackClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=demo_feedbackClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *demo_enclosing_feedback_1* updates
            if demo_enclosing_feedback_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                demo_enclosing_feedback_1.frameNStart = frameN  # exact frame index
                demo_enclosing_feedback_1.tStart = t  # local t and not account for scr refresh
                demo_enclosing_feedback_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(demo_enclosing_feedback_1, 'tStartRefresh')  # time at next scr refresh
                demo_enclosing_feedback_1.setAutoDraw(True)
            if demo_enclosing_feedback_1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > demo_enclosing_feedback_1.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    demo_enclosing_feedback_1.tStop = t  # not accounting for scr refresh
                    demo_enclosing_feedback_1.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(demo_enclosing_feedback_1, 'tStopRefresh')  # time at next scr refresh
                    demo_enclosing_feedback_1.setAutoDraw(False)
            
            # *demo_enclosing_feedback* updates
            if demo_enclosing_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                demo_enclosing_feedback.frameNStart = frameN  # exact frame index
                demo_enclosing_feedback.tStart = t  # local t and not account for scr refresh
                demo_enclosing_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(demo_enclosing_feedback, 'tStartRefresh')  # time at next scr refresh
                demo_enclosing_feedback.setAutoDraw(True)
            if demo_enclosing_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > demo_enclosing_feedback.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    demo_enclosing_feedback.tStop = t  # not accounting for scr refresh
                    demo_enclosing_feedback.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(demo_enclosing_feedback, 'tStopRefresh')  # time at next scr refresh
                    demo_enclosing_feedback.setAutoDraw(False)
            
            # *demo_target_feedback* updates
            if demo_target_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                demo_target_feedback.frameNStart = frameN  # exact frame index
                demo_target_feedback.tStart = t  # local t and not account for scr refresh
                demo_target_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(demo_target_feedback, 'tStartRefresh')  # time at next scr refresh
                demo_target_feedback.setAutoDraw(True)
            if demo_target_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > demo_target_feedback.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    demo_target_feedback.tStop = t  # not accounting for scr refresh
                    demo_target_feedback.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(demo_target_feedback, 'tStopRefresh')  # time at next scr refresh
                    demo_target_feedback.setAutoDraw(False)
            
            # *demo_fixation_feedback* updates
            if demo_fixation_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                demo_fixation_feedback.frameNStart = frameN  # exact frame index
                demo_fixation_feedback.tStart = t  # local t and not account for scr refresh
                demo_fixation_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(demo_fixation_feedback, 'tStartRefresh')  # time at next scr refresh
                demo_fixation_feedback.setAutoDraw(True)
            if demo_fixation_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > demo_fixation_feedback.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    demo_fixation_feedback.tStop = t  # not accounting for scr refresh
                    demo_fixation_feedback.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(demo_fixation_feedback, 'tStopRefresh')  # time at next scr refresh
                    demo_fixation_feedback.setAutoDraw(False)
            
            # *text_4* updates
            if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_4.frameNStart = frameN  # exact frame index
                text_4.tStart = t  # local t and not account for scr refresh
                text_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                text_4.setAutoDraw(True)
            if text_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_4.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_4.tStop = t  # not accounting for scr refresh
                    text_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text_4, 'tStopRefresh')  # time at next scr refresh
                    text_4.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in demo_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "demo_feedback"-------
        for thisComponent in demo_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        trials_2.addData('demo_enclosing_feedback_1.started', demo_enclosing_feedback_1.tStartRefresh)
        trials_2.addData('demo_enclosing_feedback_1.stopped', demo_enclosing_feedback_1.tStopRefresh)
        trials_2.addData('demo_enclosing_feedback.started', demo_enclosing_feedback.tStartRefresh)
        trials_2.addData('demo_enclosing_feedback.stopped', demo_enclosing_feedback.tStopRefresh)
        trials_2.addData('demo_target_feedback.started', demo_target_feedback.tStartRefresh)
        trials_2.addData('demo_target_feedback.stopped', demo_target_feedback.tStopRefresh)
        trials_2.addData('demo_fixation_feedback.started', demo_fixation_feedback.tStartRefresh)
        trials_2.addData('demo_fixation_feedback.stopped', demo_fixation_feedback.tStopRefresh)
        trials_2.addData('text_4.started', text_4.tStartRefresh)
        trials_2.addData('text_4.stopped', text_4.tStopRefresh)
        thisExp.nextEntry()
        
    # completed 1 repeats of 'trials_2'
    
    thisExp.nextEntry()
    
# completed 2 repeats of 'trials'


# ------Prepare to start Routine "experiment_start"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_6
gotValidClick = False  # until a click is received
# keep track of which components have finished
experiment_startComponents = [text_3, mouse_6]
for thisComponent in experiment_startComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
experiment_startClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "experiment_start"-------
while continueRoutine:
    # get current time
    t = experiment_startClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=experiment_startClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_3* updates
    if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_3.frameNStart = frameN  # exact frame index
        text_3.tStart = t  # local t and not account for scr refresh
        text_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
        text_3.setAutoDraw(True)
    # *mouse_6* updates
    if mouse_6.status == NOT_STARTED and t >= 1-frameTolerance:
        # keep track of start time/frame for later
        mouse_6.frameNStart = frameN  # exact frame index
        mouse_6.tStart = t  # local t and not account for scr refresh
        mouse_6.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_6, 'tStartRefresh')  # time at next scr refresh
        mouse_6.status = STARTED
        mouse_6.mouseClock.reset()
        prevButtonState = mouse_6.getPressed()  # if button is down already this ISN'T a new click
    if mouse_6.status == STARTED:  # only update if started and not finished!
        buttons = mouse_6.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                continueRoutine = False    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in experiment_startComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "experiment_start"-------
for thisComponent in experiment_startComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_3.started', text_3.tStartRefresh)
thisExp.addData('text_3.stopped', text_3.tStopRefresh)
# store data for thisExp (ExperimentHandler)
thisExp.addData('mouse_6.started', mouse_6.tStart)
thisExp.addData('mouse_6.stopped', mouse_6.tStop)
thisExp.nextEntry()
# the Routine "experiment_start" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
baseline_trials = data.TrialHandler(nReps=4, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('conditions_baseline.xlsx'),
    seed=None, name='baseline_trials')
thisExp.addLoop(baseline_trials)  # add the loop to the experiment
thisBaseline_trial = baseline_trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBaseline_trial.rgb)
if thisBaseline_trial != None:
    for paramName in thisBaseline_trial:
        exec('{} = thisBaseline_trial[paramName]'.format(paramName))

for thisBaseline_trial in baseline_trials:
    currentLoop = baseline_trials
    # abbreviate parameter names if possible (e.g. rgb = thisBaseline_trial.rgb)
    if thisBaseline_trial != None:
        for paramName in thisBaseline_trial:
            exec('{} = thisBaseline_trial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "baseline"-------
    continueRoutine = True
    # update component parameters for each repeat
    baseline_target.setPos((target_x, target_y))
    baseline_fixation.setOpacity(1)
    baseline_fixation.setPos((0, 0))
    # setup some python lists for storing info about the baseline_mouse
    baseline_mouse.x = []
    baseline_mouse.y = []
    baseline_mouse.leftButton = []
    baseline_mouse.midButton = []
    baseline_mouse.rightButton = []
    baseline_mouse.time = []
    gotValidClick = False  # until a click is received
    baseline_fixation.opacity = 1
    baseline_fixation.pos = [0, 0]
    mouse_center = baseline_mouse.getPos()
    first = True
    # keep track of which components have finished
    baselineComponents = [baseline_target, baseline_fixation, baseline_mouse]
    for thisComponent in baselineComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    baselineClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "baseline"-------
    while continueRoutine:
        # get current time
        t = baselineClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=baselineClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *baseline_target* updates
        if baseline_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            baseline_target.frameNStart = frameN  # exact frame index
            baseline_target.tStart = t  # local t and not account for scr refresh
            baseline_target.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(baseline_target, 'tStartRefresh')  # time at next scr refresh
            baseline_target.setAutoDraw(True)
        
        # *baseline_fixation* updates
        if baseline_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            baseline_fixation.frameNStart = frameN  # exact frame index
            baseline_fixation.tStart = t  # local t and not account for scr refresh
            baseline_fixation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(baseline_fixation, 'tStartRefresh')  # time at next scr refresh
            baseline_fixation.setAutoDraw(True)
        # *baseline_mouse* updates
        if baseline_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            baseline_mouse.frameNStart = frameN  # exact frame index
            baseline_mouse.tStart = t  # local t and not account for scr refresh
            baseline_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(baseline_mouse, 'tStartRefresh')  # time at next scr refresh
            baseline_mouse.status = STARTED
            baseline_mouse.mouseClock.reset()
            prevButtonState = baseline_mouse.getPressed()  # if button is down already this ISN'T a new click
        if baseline_mouse.status == STARTED:  # only update if started and not finished!
            x, y = baseline_mouse.getPos()
            baseline_mouse.x.append(x)
            baseline_mouse.y.append(y)
            buttons = baseline_mouse.getPressed()
            baseline_mouse.leftButton.append(buttons[0])
            baseline_mouse.midButton.append(buttons[1])
            baseline_mouse.rightButton.append(buttons[2])
            baseline_mouse.time.append(baseline_mouse.mouseClock.getTime())
        win.mouseVisible = False
        if baseline_mouse.getPressed()[0]:
            if first:
                mouse_center = baseline_mouse.getPos()
                first = False
            baseline_fixation.opacity = 0
            if euclidean_dist(baseline_mouse.getPos() - mouse_center, [0, 0]) > 0.44:
                baseline_fixation.pos = baseline_mouse.getPos() - mouse_center
                baseline_fixation.opacity = 1
                continueRoutine = False
        else:
            first = True
            baseline_fixation.opacity = 1
            baseline_fixation.pos = [0, 0]
            baseline_mouse.setPos([0, 0])
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in baselineComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "baseline"-------
    for thisComponent in baselineComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    baseline_trials.addData('baseline_target.started', baseline_target.tStartRefresh)
    baseline_trials.addData('baseline_target.stopped', baseline_target.tStopRefresh)
    baseline_trials.addData('baseline_fixation.started', baseline_fixation.tStartRefresh)
    baseline_trials.addData('baseline_fixation.stopped', baseline_fixation.tStopRefresh)
    # store data for baseline_trials (TrialHandler)
    baseline_trials.addData('baseline_mouse.x', baseline_mouse.x)
    baseline_trials.addData('baseline_mouse.y', baseline_mouse.y)
    baseline_trials.addData('baseline_mouse.leftButton', baseline_mouse.leftButton)
    baseline_trials.addData('baseline_mouse.midButton', baseline_mouse.midButton)
    baseline_trials.addData('baseline_mouse.rightButton', baseline_mouse.rightButton)
    baseline_trials.addData('baseline_mouse.time', baseline_mouse.time)
    baseline_trials.addData('baseline_mouse.started', baseline_mouse.tStart)
    baseline_trials.addData('baseline_mouse.stopped', baseline_mouse.tStop)
    baseline_fixation.opacity = 1
    baseline_fix_end_pos = baseline_fixation.pos
    # the Routine "baseline" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "baseline_feedback"-------
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    baseline_target_feedback.setPos((target_x, target_y))
    baseline_fixation_feedback.setPos(baseline_fix_end_pos)
    # keep track of which components have finished
    baseline_feedbackComponents = [baseline_target_feedback, baseline_fixation_feedback]
    for thisComponent in baseline_feedbackComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    baseline_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "baseline_feedback"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = baseline_feedbackClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=baseline_feedbackClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *baseline_target_feedback* updates
        if baseline_target_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            baseline_target_feedback.frameNStart = frameN  # exact frame index
            baseline_target_feedback.tStart = t  # local t and not account for scr refresh
            baseline_target_feedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(baseline_target_feedback, 'tStartRefresh')  # time at next scr refresh
            baseline_target_feedback.setAutoDraw(True)
        if baseline_target_feedback.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > baseline_target_feedback.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                baseline_target_feedback.tStop = t  # not accounting for scr refresh
                baseline_target_feedback.frameNStop = frameN  # exact frame index
                win.timeOnFlip(baseline_target_feedback, 'tStopRefresh')  # time at next scr refresh
                baseline_target_feedback.setAutoDraw(False)
        
        # *baseline_fixation_feedback* updates
        if baseline_fixation_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            baseline_fixation_feedback.frameNStart = frameN  # exact frame index
            baseline_fixation_feedback.tStart = t  # local t and not account for scr refresh
            baseline_fixation_feedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(baseline_fixation_feedback, 'tStartRefresh')  # time at next scr refresh
            baseline_fixation_feedback.setAutoDraw(True)
        if baseline_fixation_feedback.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > baseline_fixation_feedback.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                baseline_fixation_feedback.tStop = t  # not accounting for scr refresh
                baseline_fixation_feedback.frameNStop = frameN  # exact frame index
                win.timeOnFlip(baseline_fixation_feedback, 'tStopRefresh')  # time at next scr refresh
                baseline_fixation_feedback.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in baseline_feedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "baseline_feedback"-------
    for thisComponent in baseline_feedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    baseline_trials.addData('baseline_target_feedback.started', baseline_target_feedback.tStartRefresh)
    baseline_trials.addData('baseline_target_feedback.stopped', baseline_target_feedback.tStopRefresh)
    baseline_trials.addData('baseline_fixation_feedback.started', baseline_fixation_feedback.tStartRefresh)
    baseline_trials.addData('baseline_fixation_feedback.stopped', baseline_fixation_feedback.tStopRefresh)
    thisExp.nextEntry()
    
# completed 4 repeats of 'baseline_trials'


# ------Prepare to start Routine "break_1"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_3
gotValidClick = False  # until a click is received
# keep track of which components have finished
break_1Components = [text_2, mouse_3]
for thisComponent in break_1Components:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
break_1Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "break_1"-------
while continueRoutine:
    # get current time
    t = break_1Clock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=break_1Clock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_2* updates
    if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_2.frameNStart = frameN  # exact frame index
        text_2.tStart = t  # local t and not account for scr refresh
        text_2.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
        text_2.setAutoDraw(True)
    # *mouse_3* updates
    if mouse_3.status == NOT_STARTED and t >= 0.25-frameTolerance:
        # keep track of start time/frame for later
        mouse_3.frameNStart = frameN  # exact frame index
        mouse_3.tStart = t  # local t and not account for scr refresh
        mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
        mouse_3.status = STARTED
        mouse_3.mouseClock.reset()
        prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
    if mouse_3.status == STARTED:  # only update if started and not finished!
        buttons = mouse_3.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                continueRoutine = False    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in break_1Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "break_1"-------
for thisComponent in break_1Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_2.started', text_2.tStartRefresh)
thisExp.addData('text_2.stopped', text_2.tStopRefresh)
# store data for thisExp (ExperimentHandler)
thisExp.addData('mouse_3.started', mouse_3.tStart)
thisExp.addData('mouse_3.stopped', mouse_3.tStop)
thisExp.nextEntry()
# the Routine "break_1" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
blocks = data.TrialHandler(nReps=10, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='blocks')
thisExp.addLoop(blocks)  # add the loop to the experiment
thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
if thisBlock != None:
    for paramName in thisBlock:
        exec('{} = thisBlock[paramName]'.format(paramName))

for thisBlock in blocks:
    currentLoop = blocks
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            exec('{} = thisBlock[paramName]'.format(paramName))
    
    # set up handler to look after randomisation of conditions etc
    block_trials = data.TrialHandler(nReps=4, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('conditions_baseline.xlsx'),
        seed=None, name='block_trials')
    thisExp.addLoop(block_trials)  # add the loop to the experiment
    thisBlock_trial = block_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock_trial.rgb)
    if thisBlock_trial != None:
        for paramName in thisBlock_trial:
            exec('{} = thisBlock_trial[paramName]'.format(paramName))
    
    for thisBlock_trial in block_trials:
        currentLoop = block_trials
        # abbreviate parameter names if possible (e.g. rgb = thisBlock_trial.rgb)
        if thisBlock_trial != None:
            for paramName in thisBlock_trial:
                exec('{} = thisBlock_trial[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "rotated"-------
        continueRoutine = True
        # update component parameters for each repeat
        # setup some python lists for storing info about the rotated_mouse
        rotated_mouse.x = []
        rotated_mouse.y = []
        rotated_mouse.leftButton = []
        rotated_mouse.midButton = []
        rotated_mouse.rightButton = []
        rotated_mouse.time = []
        gotValidClick = False  # until a click is received
        if int(expInfo['participant'])%2 == 0:
            rotation = 90
        else:
            count = count + 1
            rotation = int(count/64 + 1)*10
            if rotation > 90:
                rotation = 90
        rotated_fixation.opacity = 1
        rotated_fixation.pos = [0, 0]
        first = True
        print(rotation)
        
        # keep track of which components have finished
        rotatedComponents = [rotated_target, rotated_fixation, rotated_mouse]
        for thisComponent in rotatedComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        rotatedClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "rotated"-------
        while continueRoutine:
            # get current time
            t = rotatedClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=rotatedClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rotated_target* updates
            if rotated_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rotated_target.frameNStart = frameN  # exact frame index
                rotated_target.tStart = t  # local t and not account for scr refresh
                rotated_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_target, 'tStartRefresh')  # time at next scr refresh
                rotated_target.setAutoDraw(True)
            if rotated_target.status == STARTED:  # only update if drawing
                rotated_target.setPos((target_x, target_y), log=False)
            
            # *rotated_fixation* updates
            if rotated_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rotated_fixation.frameNStart = frameN  # exact frame index
                rotated_fixation.tStart = t  # local t and not account for scr refresh
                rotated_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_fixation, 'tStartRefresh')  # time at next scr refresh
                rotated_fixation.setAutoDraw(True)
            if rotated_fixation.status == STARTED:  # only update if drawing
                rotated_fixation.setPos((0, 0), log=False)
            # *rotated_mouse* updates
            if rotated_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rotated_mouse.frameNStart = frameN  # exact frame index
                rotated_mouse.tStart = t  # local t and not account for scr refresh
                rotated_mouse.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_mouse, 'tStartRefresh')  # time at next scr refresh
                rotated_mouse.status = STARTED
                rotated_mouse.mouseClock.reset()
                prevButtonState = rotated_mouse.getPressed()  # if button is down already this ISN'T a new click
            if rotated_mouse.status == STARTED:  # only update if started and not finished!
                x, y = rotated_mouse.getPos()
                rotated_mouse.x.append(x)
                rotated_mouse.y.append(y)
                buttons = rotated_mouse.getPressed()
                rotated_mouse.leftButton.append(buttons[0])
                rotated_mouse.midButton.append(buttons[1])
                rotated_mouse.rightButton.append(buttons[2])
                rotated_mouse.time.append(rotated_mouse.mouseClock.getTime())
            win.mouseVisible = False
            if rotated_mouse.getPressed()[0]:
                if first:
                    mouse_center = rotated_mouse.getPos()
                    first = False
                rotated_fixation.opacity = 0
                if euclidean_dist(rotated_mouse.getPos()- mouse_center, [0, 0]) > 0.44:
                    mouse_pos = rotated_mouse.getPos() - mouse_center
                    rotated_fixation.pos = rotate(mouse_pos[0], mouse_pos[1], rotation)
                    rotated_fixation.opacity = 1
                    continueRoutine = False
            else:
                first = True
                rotated_fixation.opacity = 1
                rotated_fixation.pos = [0, 0]
                rotated_mouse.setPos([0, 0])
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rotatedComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "rotated"-------
        for thisComponent in rotatedComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        block_trials.addData('rotated_target.started', rotated_target.tStartRefresh)
        block_trials.addData('rotated_target.stopped', rotated_target.tStopRefresh)
        block_trials.addData('rotated_fixation.started', rotated_fixation.tStartRefresh)
        block_trials.addData('rotated_fixation.stopped', rotated_fixation.tStopRefresh)
        # store data for block_trials (TrialHandler)
        block_trials.addData('rotated_mouse.x', rotated_mouse.x)
        block_trials.addData('rotated_mouse.y', rotated_mouse.y)
        block_trials.addData('rotated_mouse.leftButton', rotated_mouse.leftButton)
        block_trials.addData('rotated_mouse.midButton', rotated_mouse.midButton)
        block_trials.addData('rotated_mouse.rightButton', rotated_mouse.rightButton)
        block_trials.addData('rotated_mouse.time', rotated_mouse.time)
        block_trials.addData('rotated_mouse.started', rotated_mouse.tStart)
        block_trials.addData('rotated_mouse.stopped', rotated_mouse.tStop)
        rotated_fixation.opacity = 1
        rotated_fix_end_pos = rotated_fixation.pos
        # the Routine "rotated" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # ------Prepare to start Routine "rotated_feedback"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        rotated_target_feedback.setPos((target_x, target_y))
        rotated_fixation_feedback.setPos(rotated_fix_end_pos)
        # keep track of which components have finished
        rotated_feedbackComponents = [rotated_target_feedback, rotated_fixation_feedback]
        for thisComponent in rotated_feedbackComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        rotated_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "rotated_feedback"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = rotated_feedbackClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=rotated_feedbackClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rotated_target_feedback* updates
            if rotated_target_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rotated_target_feedback.frameNStart = frameN  # exact frame index
                rotated_target_feedback.tStart = t  # local t and not account for scr refresh
                rotated_target_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_target_feedback, 'tStartRefresh')  # time at next scr refresh
                rotated_target_feedback.setAutoDraw(True)
            if rotated_target_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rotated_target_feedback.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    rotated_target_feedback.tStop = t  # not accounting for scr refresh
                    rotated_target_feedback.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(rotated_target_feedback, 'tStopRefresh')  # time at next scr refresh
                    rotated_target_feedback.setAutoDraw(False)
            
            # *rotated_fixation_feedback* updates
            if rotated_fixation_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rotated_fixation_feedback.frameNStart = frameN  # exact frame index
                rotated_fixation_feedback.tStart = t  # local t and not account for scr refresh
                rotated_fixation_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rotated_fixation_feedback, 'tStartRefresh')  # time at next scr refresh
                rotated_fixation_feedback.setAutoDraw(True)
            if rotated_fixation_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rotated_fixation_feedback.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    rotated_fixation_feedback.tStop = t  # not accounting for scr refresh
                    rotated_fixation_feedback.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(rotated_fixation_feedback, 'tStopRefresh')  # time at next scr refresh
                    rotated_fixation_feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rotated_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "rotated_feedback"-------
        for thisComponent in rotated_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        block_trials.addData('rotated_target_feedback.started', rotated_target_feedback.tStartRefresh)
        block_trials.addData('rotated_target_feedback.stopped', rotated_target_feedback.tStopRefresh)
        block_trials.addData('rotated_fixation_feedback.started', rotated_fixation_feedback.tStartRefresh)
        block_trials.addData('rotated_fixation_feedback.stopped', rotated_fixation_feedback.tStopRefresh)
        thisExp.nextEntry()
        
    # completed 4 repeats of 'block_trials'
    
    
    # ------Prepare to start Routine "break_2"-------
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_5
    gotValidClick = False  # until a click is received
    # keep track of which components have finished
    break_2Components = [rotated_breaks, mouse_5]
    for thisComponent in break_2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    break_2Clock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "break_2"-------
    while continueRoutine:
        # get current time
        t = break_2Clock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=break_2Clock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *rotated_breaks* updates
        if rotated_breaks.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            rotated_breaks.frameNStart = frameN  # exact frame index
            rotated_breaks.tStart = t  # local t and not account for scr refresh
            rotated_breaks.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(rotated_breaks, 'tStartRefresh')  # time at next scr refresh
            rotated_breaks.setAutoDraw(True)
        # *mouse_5* updates
        if mouse_5.status == NOT_STARTED and t >= 0.5-frameTolerance:
            # keep track of start time/frame for later
            mouse_5.frameNStart = frameN  # exact frame index
            mouse_5.tStart = t  # local t and not account for scr refresh
            mouse_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_5, 'tStartRefresh')  # time at next scr refresh
            mouse_5.status = STARTED
            mouse_5.mouseClock.reset()
            prevButtonState = mouse_5.getPressed()  # if button is down already this ISN'T a new click
        if mouse_5.status == STARTED:  # only update if started and not finished!
            buttons = mouse_5.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    continueRoutine = False        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in break_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "break_2"-------
    for thisComponent in break_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    blocks.addData('rotated_breaks.started', rotated_breaks.tStartRefresh)
    blocks.addData('rotated_breaks.stopped', rotated_breaks.tStopRefresh)
    # store data for blocks (TrialHandler)
    blocks.addData('mouse_5.started', mouse_5.tStart)
    blocks.addData('mouse_5.stopped', mouse_5.tStop)
    # the Routine "break_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    thisExp.nextEntry()
    
# completed 10 repeats of 'blocks'


# set up handler to look after randomisation of conditions etc
transfer_trials = data.TrialHandler(nReps=4, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('conditions_baseline.xlsx'),
    seed=None, name='transfer_trials')
thisExp.addLoop(transfer_trials)  # add the loop to the experiment
thisTransfer_trial = transfer_trials.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisTransfer_trial.rgb)
if thisTransfer_trial != None:
    for paramName in thisTransfer_trial:
        exec('{} = thisTransfer_trial[paramName]'.format(paramName))

for thisTransfer_trial in transfer_trials:
    currentLoop = transfer_trials
    # abbreviate parameter names if possible (e.g. rgb = thisTransfer_trial.rgb)
    if thisTransfer_trial != None:
        for paramName in thisTransfer_trial:
            exec('{} = thisTransfer_trial[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "transfer"-------
    continueRoutine = True
    # update component parameters for each repeat
    transfer_target.setPos((target_x, target_y))
    transfer_fixation.setPos((0, 0))
    # setup some python lists for storing info about the transfer_mouse
    transfer_mouse.x = []
    transfer_mouse.y = []
    transfer_mouse.leftButton = []
    transfer_mouse.midButton = []
    transfer_mouse.rightButton = []
    transfer_mouse.time = []
    gotValidClick = False  # until a click is received
    transfer_fixation.opacity = 1
    transfer_fixation.pos = [0, 0]
    first = True
    # keep track of which components have finished
    transferComponents = [transfer_target, transfer_fixation, transfer_mouse]
    for thisComponent in transferComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    transferClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "transfer"-------
    while continueRoutine:
        # get current time
        t = transferClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=transferClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *transfer_target* updates
        if transfer_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            transfer_target.frameNStart = frameN  # exact frame index
            transfer_target.tStart = t  # local t and not account for scr refresh
            transfer_target.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(transfer_target, 'tStartRefresh')  # time at next scr refresh
            transfer_target.setAutoDraw(True)
        
        # *transfer_fixation* updates
        if transfer_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            transfer_fixation.frameNStart = frameN  # exact frame index
            transfer_fixation.tStart = t  # local t and not account for scr refresh
            transfer_fixation.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(transfer_fixation, 'tStartRefresh')  # time at next scr refresh
            transfer_fixation.setAutoDraw(True)
        # *transfer_mouse* updates
        if transfer_mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            transfer_mouse.frameNStart = frameN  # exact frame index
            transfer_mouse.tStart = t  # local t and not account for scr refresh
            transfer_mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(transfer_mouse, 'tStartRefresh')  # time at next scr refresh
            transfer_mouse.status = STARTED
            transfer_mouse.mouseClock.reset()
            prevButtonState = transfer_mouse.getPressed()  # if button is down already this ISN'T a new click
        if transfer_mouse.status == STARTED:  # only update if started and not finished!
            x, y = transfer_mouse.getPos()
            transfer_mouse.x.append(x)
            transfer_mouse.y.append(y)
            buttons = transfer_mouse.getPressed()
            transfer_mouse.leftButton.append(buttons[0])
            transfer_mouse.midButton.append(buttons[1])
            transfer_mouse.rightButton.append(buttons[2])
            transfer_mouse.time.append(transfer_mouse.mouseClock.getTime())
        win.mouseVisible = False
        if transfer_mouse.getPressed()[0]:
            if first:
                mouse_center = transfer_mouse.getPos()
                first = False
            transfer_fixation.opacity = 0
            if euclidean_dist(transfer_mouse.getPos()- mouse_center, [0, 0]) > 0.44:
                transfer_fixation.pos = transfer_mouse.getPos() - mouse_center
                transfer_fixation.opacity = 1
                continueRoutine = False
        else:
            first = False
            transfer_fixation.opacity = 1
            transfer_fixation.pos = [0, 0]
            transfer_mouse.setPos([0, 0])
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in transferComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "transfer"-------
    for thisComponent in transferComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    transfer_trials.addData('transfer_target.started', transfer_target.tStartRefresh)
    transfer_trials.addData('transfer_target.stopped', transfer_target.tStopRefresh)
    transfer_trials.addData('transfer_fixation.started', transfer_fixation.tStartRefresh)
    transfer_trials.addData('transfer_fixation.stopped', transfer_fixation.tStopRefresh)
    # store data for transfer_trials (TrialHandler)
    transfer_trials.addData('transfer_mouse.x', transfer_mouse.x)
    transfer_trials.addData('transfer_mouse.y', transfer_mouse.y)
    transfer_trials.addData('transfer_mouse.leftButton', transfer_mouse.leftButton)
    transfer_trials.addData('transfer_mouse.midButton', transfer_mouse.midButton)
    transfer_trials.addData('transfer_mouse.rightButton', transfer_mouse.rightButton)
    transfer_trials.addData('transfer_mouse.time', transfer_mouse.time)
    transfer_trials.addData('transfer_mouse.started', transfer_mouse.tStart)
    transfer_trials.addData('transfer_mouse.stopped', transfer_mouse.tStop)
    transfer_fixation.opacity = 1
    transfer_fix_end_pos = transfer_fixation.pos
    # the Routine "transfer" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # ------Prepare to start Routine "transfer_feedback"-------
    continueRoutine = True
    routineTimer.add(0.500000)
    # update component parameters for each repeat
    transfer_target_feedback.setPos((target_x, target_y))
    transfer_fixation_feedback.setPos(transfer_fix_end_pos)
    # keep track of which components have finished
    transfer_feedbackComponents = [transfer_target_feedback, transfer_fixation_feedback]
    for thisComponent in transfer_feedbackComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    transfer_feedbackClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "transfer_feedback"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = transfer_feedbackClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=transfer_feedbackClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *transfer_target_feedback* updates
        if transfer_target_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            transfer_target_feedback.frameNStart = frameN  # exact frame index
            transfer_target_feedback.tStart = t  # local t and not account for scr refresh
            transfer_target_feedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(transfer_target_feedback, 'tStartRefresh')  # time at next scr refresh
            transfer_target_feedback.setAutoDraw(True)
        if transfer_target_feedback.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > transfer_target_feedback.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                transfer_target_feedback.tStop = t  # not accounting for scr refresh
                transfer_target_feedback.frameNStop = frameN  # exact frame index
                win.timeOnFlip(transfer_target_feedback, 'tStopRefresh')  # time at next scr refresh
                transfer_target_feedback.setAutoDraw(False)
        
        # *transfer_fixation_feedback* updates
        if transfer_fixation_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            transfer_fixation_feedback.frameNStart = frameN  # exact frame index
            transfer_fixation_feedback.tStart = t  # local t and not account for scr refresh
            transfer_fixation_feedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(transfer_fixation_feedback, 'tStartRefresh')  # time at next scr refresh
            transfer_fixation_feedback.setAutoDraw(True)
        if transfer_fixation_feedback.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > transfer_fixation_feedback.tStartRefresh + 0.5-frameTolerance:
                # keep track of stop time/frame for later
                transfer_fixation_feedback.tStop = t  # not accounting for scr refresh
                transfer_fixation_feedback.frameNStop = frameN  # exact frame index
                win.timeOnFlip(transfer_fixation_feedback, 'tStopRefresh')  # time at next scr refresh
                transfer_fixation_feedback.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in transfer_feedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "transfer_feedback"-------
    for thisComponent in transfer_feedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    transfer_trials.addData('transfer_target_feedback.started', transfer_target_feedback.tStartRefresh)
    transfer_trials.addData('transfer_target_feedback.stopped', transfer_target_feedback.tStopRefresh)
    transfer_trials.addData('transfer_fixation_feedback.started', transfer_fixation_feedback.tStartRefresh)
    transfer_trials.addData('transfer_fixation_feedback.stopped', transfer_fixation_feedback.tStopRefresh)
    thisExp.nextEntry()
    
# completed 4 repeats of 'transfer_trials'


# ------Prepare to start Routine "Done"-------
continueRoutine = True
# update component parameters for each repeat
# setup some python lists for storing info about the mouse_4
gotValidClick = False  # until a click is received
# keep track of which components have finished
DoneComponents = [done_text, mouse_4]
for thisComponent in DoneComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
DoneClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "Done"-------
while continueRoutine:
    # get current time
    t = DoneClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=DoneClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *done_text* updates
    if done_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        done_text.frameNStart = frameN  # exact frame index
        done_text.tStart = t  # local t and not account for scr refresh
        done_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(done_text, 'tStartRefresh')  # time at next scr refresh
        done_text.setAutoDraw(True)
    # *mouse_4* updates
    if mouse_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        mouse_4.frameNStart = frameN  # exact frame index
        mouse_4.tStart = t  # local t and not account for scr refresh
        mouse_4.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(mouse_4, 'tStartRefresh')  # time at next scr refresh
        mouse_4.status = STARTED
        mouse_4.mouseClock.reset()
        prevButtonState = mouse_4.getPressed()  # if button is down already this ISN'T a new click
    if mouse_4.status == STARTED:  # only update if started and not finished!
        buttons = mouse_4.getPressed()
        if buttons != prevButtonState:  # button state changed?
            prevButtonState = buttons
            if sum(buttons) > 0:  # state changed to a new click
                continueRoutine = False    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in DoneComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "Done"-------
for thisComponent in DoneComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('done_text.started', done_text.tStartRefresh)
thisExp.addData('done_text.stopped', done_text.tStopRefresh)
# store data for thisExp (ExperimentHandler)
thisExp.addData('mouse_4.started', mouse_4.tStart)
thisExp.addData('mouse_4.stopped', mouse_4.tStop)
thisExp.nextEntry()
# the Routine "Done" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
