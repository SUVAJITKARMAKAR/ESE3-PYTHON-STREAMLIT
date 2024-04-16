# IMPORTING THE REQUIRED MODULES IN THE WORK DIRECTORY
import streamlit as stream
import pandas as panda
import numpy as num
import matplotlib.pyplot as plotting
import seaborn as sea

# SETTING THE PAGE CONFIGURATION
stream.set_page_config(
    page_title="VISUALIZATION",
    page_icon=":rocket:",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    #SETTING UP THE SIDEBAR 
    stream.sidebar.header("DATA VISUALIZATION OPTIONS")

    # PAGE TITLE
    stream.title('DATA PROCESSING AND VISUALTIZATION')

    # FILE UPLOAD
    uploaded_file = stream.file_uploader("UPLOAD CSV FILE FORMAT", type=["csv"])

    if uploaded_file is not None:

        dframe = panda.read_csv(uploaded_file)
    
        # DISPLLAYING THE RAW DATA
        stream.subheader('RAW DATA')
        stream.write(dframe)
    
        # SELECTING THE PRE-PROCESSING OPTIONS
        stream.subheader('PREPROCESSING OPTIONS')
    
        # CHECKING FOR MISSING VALUES
        if stream.checkbox('CHECK FOR MISSING VALUES IN THE DATASET'):
            stream.subheader('MISSING VALUES')
            stream.write(dframe.isnull().sum())
    
    
        # DATA VISUALIZATION OPTIONS
        stream.subheader('DATA VISUALIZATION OPTIONS')
    
        # 2D PLOTTING OPTIONS
        stream.sidebar.subheader('2D PLOTTING OPTIONS')
        x_axis_2d = stream.sidebar.selectbox('SELECT X-AXIS', dframe.columns, key='select_x_axis_2d')
        y_axis_2d = stream.sidebar.selectbox('SELECT Y-AXIS', dframe.columns, key='select_y_axis_2d')
        plot_type_2d = stream.sidebar.selectbox('SELECT PLOTTING TYPE', ['Violin Plot', 'Box Plot', 'Scatter Plot'])

        # 3D PLOTTING OPTIONS
        stream.sidebar.subheader('3D PLOTTING OPTIONS')
        plot_type_3d = stream.sidebar.selectbox('SELECT  PLOTTING TYPE', ['Scatter Plot', 'Surface Plot', 'Stem Plot'])

        if plot_type_2d == 'Violin Plot':
            fig, ax = plotting.subplots()
            sea.violinplot(x=dframe[x_axis_2d], y=dframe[y_axis_2d], ax=ax)
            ax.set_xlabel(x_axis_2d)
            ax.set_ylabel(y_axis_2d)
            stream.pyplot(fig)
        elif plot_type_2d == 'Box Plot':
            fig, ax = plotting.subplots()
            sea.boxplot(x=dframe[x_axis_2d], y=dframe[y_axis_2d], ax=ax)
            ax.set_xlabel(x_axis_2d)
            ax.set_ylabel(y_axis_2d)
            stream.pyplot(fig)
        elif plot_type_2d == 'Scatter Plot':
            fig, ax = plotting.subplots()
            ax.scatter(dframe[x_axis_2d], dframe[y_axis_2d])
            ax.set_xlabel(x_axis_2d)
            ax.set_ylabel(y_axis_2d)
            stream.pyplot(fig)
    
        if plot_type_3d == 'Scatter Plot':
            x_axis_3d = stream.sidebar.selectbox('SELECT X-AXIS', dframe.columns, key='select_x_axis_3d_scatter')
            y_axis_3d = stream.sidebar.selectbox('SELECT Y-AXIS', dframe.columns, key='select_y_axis_3d_scatter')
            z_axis_3d = stream.sidebar.selectbox('SELECT Z-AXIS', dframe.columns, key='select_z_axis_3d_scatter')
            fig = plotting.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(dframe[x_axis_3d], dframe[y_axis_3d], dframe[z_axis_3d])
            ax.set_xlabel(x_axis_3d)
            ax.set_ylabel(y_axis_3d)
            ax.set_zlabel(z_axis_3d)
            stream.pyplot(fig)
        elif plot_type_3d == 'Surface Plot':
            x_axis_3d = stream.sidebar.selectbox('SELECT X-AXIS', dframe.columns, key='select_x_axis_3d_surface')
            y_axis_3d = stream.sidebar.selectbox('SELECT Y-AXIS', dframe.columns, key='select_y_axis_3d_surface')
            z_axis_3d = stream.sidebar.selectbox('SELECT Z-AXIS', dframe.columns, key='select_z_axis_3d_surface')
            fig = plotting.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(dframe[x_axis_3d], dframe[y_axis_3d], dframe[z_axis_3d], cmap='viridis')
            ax.set_xlabel(x_axis_3d)
            ax.set_ylabel(y_axis_3d)
            ax.set_zlabel(z_axis_3d)
            stream.pyplot(fig)
        elif plot_type_3d == 'Stem Plot':
            x_axis_3d = stream.sidebar.selectbox('SELECT X-AXIS', dframe.columns, key='select_x_axis_3d_stem')
            y_axis_3d = stream.sidebar.selectbox('SELECT Y-AXIS', dframe.columns, key='select_y_axis_3d_stem')
            z_axis_3d = stream.sidebar.selectbox('SELECT Z-AXIS', dframe.columns, key='select_z_axis_3d_stem')
        
        fig = plotting.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # SCATTER PLOTTING
        ax.scatter(dframe[x_axis_3d], dframe[y_axis_3d], dframe[z_axis_3d], c='b', marker='o')
        
        # STEM LINES
        for i in range(len(dframe)):
            ax.plot([dframe[x_axis_3d][i], dframe[x_axis_3d][i]], 
                    [dframe[y_axis_3d][i], dframe[y_axis_3d][i]], 
                    [0, dframe[z_axis_3d][i]], 
                    color='g')
        
        ax.set_xlabel(x_axis_3d)
        ax.set_ylabel(y_axis_3d)
        ax.set_zlabel(z_axis_3d)
        
        stream.pyplot(fig)

# MAIN OPERATION INITIALITION POINT 
if __name__ == '__main__':
    main()
