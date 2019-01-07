import numpy as np
import matplotlib.pyplot as plt
import operations as op
import math as ma

key = 'ground';
name1 = key + '_trajectory.txt';#You should change this to the file name of your trajectory file
name2 = 'ave' + key + 'traj.txt';
#name3 = key + '_traj.png'
file = open(name1,'r');
raw0 = file.readlines();
file.close();

#raw = raw0[48*1000:48*11000];

lattice = np.zeros((3,3));
lattice[0][0] = 8.58;
lattice[1][1] = 8.87;
lattice[2][2] = 12.63;

length = len(raw);
numStruct = int(length/48);
allData = np.zeros((numStruct,48,3));

all_h = np.zeros((numStruct,24,3));
all_c = np.zeros((numStruct,4,3));
all_n = np.zeros((numStruct,4,3));
all_pb = np.zeros((numStruct,4,3));
all_i = np.zeros((numStruct,12,3));

count_h = 0;
count_c = 0;
count_n = 0;
count_pb = 0;
count_i  = 0;

for i in range(length):
    temp = raw[i].split();

    indexIndividual = int(np.floor(i/48));
    if temp[0] == 'H':
        index_h = count_h%24;
        all_h[indexIndividual][index_h] = temp[1:4];
        count_h += 1;
        
    elif temp[0] == 'C':
        index_c = count_c%4;
        all_c[indexIndividual][index_c] = temp[1:4];
        count_c += 1;
        
    elif temp[0] == 'N':
        index_n = count_n%4;
        all_n[indexIndividual][index_n] = temp[1:4];
        count_n += 1;
        
    elif temp[0] == 'Pb':
        index_pb = count_pb%4;
        all_pb[indexIndividual][index_pb] = temp[1:4];
        count_pb += 1;
        
    elif temp[0] == 'I':
        index_i = count_i%12;
        all_i[indexIndividual][index_i] = temp[1:4];     
        count_i += 1;
    
    index = i%48;
    allData[indexIndividual][index] = temp[1:4];

'''Pull back all the atoms according its previous configuration'''
diff = np.zeros((48,3));
for i in range(1,numStruct,1):
    diff = allData[i] - allData[i-1];
    diff_h = all_h[i] - all_h[i-1];
    diff_c = all_c[i] - all_c[i-1];
    diff_n = all_n[i] - all_n[i-1];
    diff_pb = all_pb[i] - all_pb[i-1];
    diff_i = all_i[i] - all_i[i-1];
    for j in range(48):
        for k in range(3):
            diff[j][k] = round(diff[j][k]/lattice[k][k]);
        allData[i][j] = allData[i][j] - np.matmul(diff[j],lattice);
        
    for j in range(24):
        for k in range(3):
            diff_h[j][k] = round(diff_h[j][k]/lattice[k][k]);
        all_h[i][j] = all_h[i][j] - np.matmul(diff_h[j],lattice);
        
    for j in range(4):
        for k in range(3):
            diff_c[j][k] = round(diff_c[j][k]/lattice[k][k]);
        all_c[i][j] = all_c[i][j] - np.matmul(diff_c[j],lattice);
        
    for j in range(4):
        for k in range(3):
            diff_n[j][k] = round(diff_n[j][k]/lattice[k][k]);
        all_n[i][j] = all_n[i][j] - np.matmul(diff_n[j],lattice);
        
    for j in range(4):
        for k in range(3):
            diff_pb[j][k] = round(diff_pb[j][k]/lattice[k][k]);
        all_pb[i][j] = all_pb[i][j] - np.matmul(diff_pb[j],lattice);
        
    for j in range(12):
        for k in range(3):
            diff_i[j][k] = round(diff_i[j][k]/lattice[k][k]);
        all_i[i][j] = all_i[i][j] - np.matmul(diff_i[j],lattice);

'''Get the rotation of each molecule. phi, cos(theta)'''
molecules = np.zeros((numStruct,4,2));
for i in range(numStruct):
    for j in range(4):
        tempN = np.zeros((3,2));
        op.gothrough(all_c[i][j],all_n[i],tempN);
        tempMolecule = all_n[i][int(tempN[0][0])] - all_c[i][j];
        tempMolecule = tempMolecule/np.linalg.norm(tempMolecule);
        molecules[i][j][1] = tempMolecule[2];
        x = tempMolecule[0]; y = tempMolecule[1];
        if tempMolecule[1] > 0:
            molecules[i][j][0] = ma.acos(x/(x**2+y**2)**0.5)/ma.pi*180;
        else:
            molecules[i][j][0] = 360 - ma.acos(x/(x**2+y**2)**0.5)/ma.pi*180;

fig1 = plt.figure(figsize=(20,16));
mole1 = fig1.add_subplot(2,2,1);
mole1.plot(molecules[:,0,0]);
plt.title('Molecule 1',fontsize=20)
plt.ylabel(r'$\phi$',fontsize=20)
plt.xlabel('time/fs',fontsize=20)

mole2 = fig1.add_subplot(2,2,2);
mole2.plot(molecules[:,1,0]);
plt.title('Molecule 2',fontsize=20)
plt.ylabel(r'$\phi$',fontsize=20)
plt.xlabel('time/fs',fontsize=20)

mole3 = fig1.add_subplot(2,2,3);
mole3.plot(molecules[:,2,0]);
plt.title('Molecule 3',fontsize=20)
plt.ylabel(r'$\phi$',fontsize=20)
plt.xlabel('time/fs',fontsize=20)

mole4 = fig1.add_subplot(2,2,4);
mole4.plot(molecules[:,3,0]);
plt.title('Molecule 4',fontsize=20)
plt.ylabel(r'$\phi$',fontsize=20)
plt.xlabel('time/fs',fontsize=20)

#fig1.savefig(key+'_phi_vs_time.png',fontsize=20)

fig2 = plt.figure(figsize=(20,16));
mole1 = fig2.add_subplot(2,2,1);
mole1.plot(molecules[:,0,1]);
plt.title('Molecule 1',fontsize=20)
plt.ylabel(r'cos$\theta$',fontsize=20)
plt.xlabel('time/fs',fontsize=20)

mole2 = fig2.add_subplot(2,2,2);
mole2.plot(molecules[:,1,1]);
plt.title('Molecule 2',fontsize=20)
plt.ylabel(r'cos$\theta$',fontsize=20)
plt.xlabel('time/fs',fontsize=20)

mole3 = fig2.add_subplot(2,2,3);
mole3.plot(molecules[:,2,1]);
plt.title('Molecule 3',fontsize=20)
plt.ylabel(r'cos$\theta$',fontsize=20)
plt.xlabel('time/fs',fontsize=20)

mole4 = fig2.add_subplot(2,2,4);
mole4.plot(molecules[:,3,1]);
plt.title('Molecule 4',fontsize=20)
plt.ylabel(r'cos$\theta$',fontsize=20)
plt.xlabel('time/fs',fontsize=20)

#fig2.savefig(key + '_costheta_vs_time.png')
'''End of analyzing the rotation of the molecule'''

'''Plot the displacement'''
'''Calculate the displacement of atomic movement'''
#dispH = np.zeros((numStruct,24));
#dispC = np.zeros((numStruct,4));
#dispN = np.zeros((numStruct,4));
#dispPb = np.zeros((numStruct,4));
#dispI = np.zeros((numStruct,12));
#
#for i in range(1,numStruct,1):
#    for j in range(24):
#        dispH[i][j] = np.linalg.norm(all_h[i][j] - all_h[0][j]);
#
#    for j in range(4):
#        dispC[i][j] = np.linalg.norm(all_c[i][j] - all_c[0][j]);
#        
#    for j in range(4):
#        dispN[i][j] = np.linalg.norm(all_n[i][j] - all_n[0][j]);
#        
#    for j in range(4):
#        dispPb[i][j] = np.linalg.norm(all_pb[i][j] - all_pb[0][j]);
#        
#    for j in range(12):
#        dispI[i][j] = np.linalg.norm(all_i[i][j] - all_i[0][j]);
#        
#        
#'''Plot the single atom displacement'''
#fig = plt.figure(figsize=(20,10));
#
#index = 0;
#
#H1 = fig.add_subplot(2,3,1);
#H1.plot(dispH[:,index*6]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#H2 = fig.add_subplot(2,3,2);
#H2.plot(dispH[:,1+index*6]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#H3 = fig.add_subplot(2,3,3);
#H3.plot(dispH[:,2+index*6]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#H4 = fig.add_subplot(2,3,4);
#H4.plot(dispH[:,3+index*6]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#H5 = fig.add_subplot(2,3,5);
#H5.plot(dispH[:,4+index*6]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#H6 = fig.add_subplot(2,3,6);
#H6.plot(dispH[:,5+index*6]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#nameH = 'H_' + str(6*index+1) + '_' + str(6*(index+1)) + '.png';
#fig.savefig(nameH);

'''Other atoms'''
#fig = plt.figure(figsize=(20,16));
#
#index = 0;
#
#C1 = fig.add_subplot(2,2,1);
#C1.plot(dispPb[:,index*4]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#C2 = fig.add_subplot(2,2,2);
#C2.plot(dispPb[:,1+index*4]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#C3 = fig.add_subplot(2,2,3);
#C3.plot(dispPb[:,2+index*4]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#C4 = fig.add_subplot(2,2,4);
#C4.plot(dispPb[:,3+index*4]);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#
#nameC = 'Pb_' + str(4*index+1) + '_' + str(4*(index+1)) + '.png';
#fig.savefig(nameC);


'''End of it'''

#aveH = np.average(dispH, axis=1);
#aveC = np.average(dispC, axis=1);
#aveN = np.average(dispN, axis=1);
#avePb = np.average(dispPb, axis=1);
#aveI = np.average(dispI, axis=1);
#
#
#output = open(name2,'w');
#for i in range(numStruct):
#    item = str(float(aveH[i])) + ' ' + str(float(aveC[i])) + ' ' + str(float(aveN[i])) + ' ' +\
#    str(float(avePb[i])) + ' ' + str(float(aveI[i])) + '\n'; 
#    output.write(item);
#
#output.close();
#
#filePlot = open(name2,'r');
#plotLines = filePlot.readlines();
#filePlot.close();
#aveH0 = np.zeros((numStruct,1));
#aveC0 = np.zeros((numStruct,1));
#aveN0 = np.zeros((numStruct,1));
#avePb0 = np.zeros((numStruct,1));
#aveI0 = np.zeros((numStruct,1));
#for i in range(numStruct):
#    temp = plotLines[i].split();
#    aveH0[i] = float(temp[0]);
#    aveC0[i] = float(temp[1]);
#    aveN0[i] = float(temp[2]);
#    avePb0[i] = float(temp[3]);
#    aveI0[i] = float(temp[4]);
#
#fig = plt.figure(figsize=(20, 20))
#pltH = fig.add_subplot(3,2,1);
#pltH.plot(aveH0);
#plt.title('H',fontsize = 20);
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#plt.ylim(0,2.0);
#
#pltC = fig.add_subplot(3,2,2);
#pltC.plot(aveC0);
#plt.title('C',fontsize = 20)
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#plt.ylim(0,0.7);
#
#pltN = fig.add_subplot(3,2,3);
#pltN.plot(aveN0);
#plt.title('N',fontsize = 20)
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#plt.ylim(0,0.7);
#
#pltPb = fig.add_subplot(3,2,4);
#pltPb.plot(avePb0);
#plt.title('Pb',fontsize = 20)
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#plt.ylim(0,0.7);
#
#pltI = fig.add_subplot(3,2,5);
#pltI.plot(aveI0);
#plt.title('I',fontsize = 20)
#plt.xlabel('time/fs',fontsize = 20)
#plt.ylabel(r'$\delta$x',fontsize = 20)
#plt.ylim(0,0.7);
#
#fig.savefig(name3)
'''End of plotting the displacement'''