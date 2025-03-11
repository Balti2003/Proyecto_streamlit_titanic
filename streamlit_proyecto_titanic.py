import streamlit as st
import pandas as pd
import copy
import plotly.express as px
from titanic_ai_model import get_model

st.set_page_config(
    page_title = "Proyecto Titanic",
    page_icon = "🗏",
    layout = "wide",
    initial_sidebar_state = "expanded",
)

if "df" not in st.session_state:
    st.session_state["df"] = pd.DataFrame()

if "df_fil" not in st.session_state:
    st.session_state["df_fil"] = pd.DataFrame()

#Funciones
def clean_data(df):
    survived_dict = {1: "Yes", 0: "No"}
    pclass_dict = {1: "1st", 2: "2nd", 3: "3rd"}
    embarkment_dict = {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
    
    df.replace({"survived": survived_dict}, inplace = True)
    df.replace({"pclass": pclass_dict}, inplace = True)
    df.replace({"embarked": embarkment_dict}, inplace = True)
    
    df.dropna(subset = ["fare"], inplace = True)
    df.dropna(subset = ["age"], inplace = True)
    df.dropna(subset = ["embarked"], inplace = True)
    df.dropna(subset = ["cabin"], inplace = True)
    
    df["count"] = 1
    
    return df

@st.cache_data
def get_data(file):
    df = pd.read_csv(file)
    df = clean_data(df)
    
    return df

@st.cache_data
def get_values(column):
    return sorted(st.session_state['df'][column].dropna().unique())

def update_df():   
    st.session_state['df_fil'] = st.session_state['df'][
        (st.session_state["df"]["survived"].isin(st.session_state["survived"])) &
        (st.session_state["df"]["pclass"].isin(st.session_state["pclass"])) &
        (st.session_state["df"]["sex"].isin(st.session_state["sex"])) &
        (st.session_state["df"]["embarked"].isin(st.session_state["embarked"])) & 
        ((st.session_state["df"]["age"] >= st.session_state["age"][0]) & (st.session_state["df"]["age"] <= st.session_state["age"][1])) &
        ((st.session_state["df"]["sibsp"] >= st.session_state["sibsp"][0]) & (st.session_state["df"]["sibsp"] <= st.session_state["sibsp"][1])) &
        ((st.session_state["df"]["parch"] >= st.session_state["parch"][0]) & (st.session_state["df"]["parch"] <= st.session_state["parch"][1])) &
        ((st.session_state["df"]["fare"] >= st.session_state["fare"][0]) & (st.session_state["df"]["fare"] <= st.session_state["fare"][1])) 
    ]
    
def generate_plot(var1, var2, var3, num_var, color_var, plot_type):
    if plot_type == "Bar":
        fig = px.bar(
            st.session_state["df_fil"],
            x = var1,
            y = num_var,
            color = color_var,
        )
    elif plot_type == "Pie":
        fig = px.pie(
            st.session_state["df_fil"],
            values = num_var,
            names = var1,
        )
    elif plot_type == "Scatter":
        fig = px.scatter(
            st.session_state["df_fil"],
            x = var1,
            y = var2,
            size = num_var,
            color = color_var,
        )
    elif plot_type == "Heatmap":
        fig = px.density_heatmap(
            st.session_state["df_fil"],
            x = var1,
            y = var2,
            z = num_var,
            text_auto = True, 
        )
    elif plot_type == "Treemap":
        fig = px.treemap(
            st.session_state["df_fil"],
            path = [var1, var2, var3],
            values = num_var,
            color = color_var,
        )
        
    return fig
    
#Fin Funciones
st.title("Titanic passenger data")

st.session_state["df"] = get_data("./train.csv")

def page1():
    st.subheader("🚢 Data description")
    
    st.markdown(
        open(r"./titanic_table_description.html").read(), 
        unsafe_allow_html=True
    )
    with st.expander("Dataframe"):
        st.write(st.session_state["df"])
    
    with st.expander("Describe"):
        st.write(st.session_state["df"].describe())
        
def page2(): 
    if st.session_state["df_fil"].empty:
        st.session_state["df_fil"] = copy.copy(st.session_state["df"])
    
    st.subheader("📊 Data analytics")
    
    col_plot_1, col_plot_2 = st.columns([5,1])
    
    with col_plot_1:
        fig_plot =  st.empty()
        
    with col_plot_2:
        plot_type = st.selectbox(
            "Type of plot",
            options = ["Bar", "Pie", "Scatter", "Heatmap", "Treemap"]
        )
        
        var1 = st.selectbox(
            "1st variable",
            options = st.session_state["df_fil"].columns, 
        )
        
        num_var = st.selectbox (
            "Numeric variable",
            options = ["count", "fare", "age", "sibsp", "parch"],
        )
        
        color_var = st.selectbox(
            "Color variable",
            options = st.session_state["df_fil"].columns,
        )
        
        var2 = st.selectbox(
            "2nd variable",
            options = st.session_state["df_fil"].columns,
        )
        
        var3 = st.selectbox(
            "3rd variable",
            options = st.session_state["df_fil"].columns,
        )
    
    with st.expander("Filters"):    
        with st.form(key="filter_form"):
            
            col_fil_1, col_fil_2 = st.columns([1, 1]) 
    
            with col_fil_1:
                survived_values = get_values("survived")
                st.multiselect(
                    "Survived",
                    options = survived_values,
                    help = "Select survived",
                    default = survived_values,
                    key = "survived"
                )
                
                pclass_values = get_values("pclass")
                st.multiselect(
                    "Pclass",
                    options = pclass_values,
                    help = "Select pclass",
                    default = pclass_values,
                    key = "pclass"
                )
                
                sex_values = get_values("sex")
                st.multiselect(
                    "Sex",
                    options = sex_values,
                    help = "Select sex",
                    default = sex_values,
                    key = "sex"
                )
                
                embark_values = get_values("embarked")
                st.multiselect(
                    "Embarked",
                    options = embark_values,
                    help = "Select embarked",
                    default = embark_values,
                    key = "embarked"
                )
    
            with col_fil_2:
        
                age_values = get_values("age")
                st.slider(
                    "Age",
                    min_value = min(age_values),
                    max_value = max(age_values),
                    value = [min(age_values), max(age_values)],
                    key = "age"
                )
                
                sibsp_values = get_values("sibsp")
                st.slider(
                    "Sibsp",
                    min_value = min(sibsp_values),
                    max_value = max(sibsp_values),
                    value = [min(sibsp_values), max(sibsp_values)],
                    key = "sibsp"
                )
                
                parch_values = get_values("parch")
                st.slider(
                    "Parch",
                    min_value = min(parch_values),
                    max_value = max(parch_values),
                    value = [min(parch_values), max(parch_values)],
                    key = "parch"
                )
                
                fare_values = get_values("fare")
                st.slider(
                    "Fare",
                    min_value = min(fare_values),
                    max_value = max(fare_values),
                    value = [min(fare_values), max(fare_values)],
                    key = "fare"
                )
                
            submit = st.form_submit_button("Update")

        if submit:
            update_df()

    fig = generate_plot(var1, var2, var3, num_var, color_var, plot_type)
    fig_plot.write(fig)
   
def page3():
    st.subheader("🤖 Artificial Intelligence")
    
    model = get_model()
    
    with st.form("prediction form"):
        
        col_pred_1, col_pred_2, col_pred_3 = st.columns([1,1,1])
    
        with col_pred_1:
            class_input = st.selectbox(
                    "Class",
                    options = [1,2,3]
                    )
                    
            age_input = st.number_input(
                    "Age",
                    min_value = 0.0,
                    max_value = 100.0,
                    )
                    
            sibsp_input = st.number_input(
                    "Siblings",
                    min_value = 0,
                    max_value = 10,
                    )
                    
        with col_pred_2:
            parch_input = st.number_input(
                    "Parents/Children",
                    min_value = 0,
                    max_value = 10,
                    )
                    
            fare_input = st.number_input(
                    "Fare",
                    min_value = 0.0,
                    max_value = 1000.0,
                    )
        
        with col_pred_3:        
            sex_input = st.toggle(
                    "Sex male",
                    )
                    
            q_input = st.toggle(
                    "Queenstown embarked",
                    )
                    
            s_input = st.toggle(
                    "Southhampton embarked",
                    )

            submit_prediction = st.form_submit_button("Submit prediction")
        
        if submit_prediction:
            
            input_vector = [[
                class_input,
                age_input,
                sibsp_input,
                parch_input,
                fare_input,
                sex_input,
                q_input,
                s_input,
            ]]
            
            y_pred = model.predict(input_vector)

            if y_pred:
                st.success("The passenger is likely to survive!")
            else:
                st.error("The passenger is likely to die...")
        
pg = st.navigation(
    {"D&A": [
        st.Page(page1, title="Data description", icon="🚢"),
        st.Page(page2, title="Data analytics", icon="📊"),
    ],
    
    "AI": [
        st.Page(page3, title="Artificial Intelligence", icon="🤖"),
    ],
    }
)
pg.run()

