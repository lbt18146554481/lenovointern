import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyod.models.lof import LOF
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.ocsvm import OCSVM
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.lscp import LSCP
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.models.vae import VAE
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.so_gaal import SO_GAAL
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 设置页面配置
st.set_page_config(page_title="分类任务机器学习平台", layout="wide")

# 添加自动滚动到顶部的JavaScript代码
js_code = """
<script>
    window.scrollTo(0, 0);
</script>
"""
st.markdown(js_code, unsafe_allow_html=True)

# 初始化会话状态
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'outliers' not in st.session_state:
    st.session_state.outliers = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = 0

# 定义页面标题
pages = ["特征分析", "异常值检测", "模型训练"]

# 主标题


# 显示当前页面标题
st.header(f"当前步骤：{pages[st.session_state.current_page]}")

# 文件上传（在所有页面都显示）
uploaded_file = st.file_uploader("上传数据文件", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        with st.spinner('正在加载数据...'):
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
        
        st.session_state.data = data
        st.success("数据上传成功！")

        # 页面内容
        if st.session_state.current_page == 0:
            # 特征分析页面
            # 显示数据预览
            st.subheader("数据预览")
            st.dataframe(data.head())
            
            # 显示数据基本信息
            st.subheader("数据基本信息")
            col1, col2, col3 = st.columns(3)
            col1.metric("行数", data.shape[0])
            col2.metric("列数", data.shape[1])
            col3.metric("特征数", data.shape[1])
            
            # 单特征分析
            st.subheader("单特征分析")
            selected_feature = st.selectbox("选择要分析的特征", data.columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 箱线图
                fig_box = go.Figure()
                fig_box.add_trace(go.Box(y=data[selected_feature], name=selected_feature))
                fig_box.update_layout(title="箱线图")
                st.plotly_chart(fig_box)
            
            with col2:
                # 分布图
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=data[selected_feature], nbinsx=30))
                fig_dist.update_layout(title="分布图")
                st.plotly_chart(fig_dist)
            
            # 统计信息
            st.subheader("统计信息")
            stats = data[selected_feature].describe()
            st.dataframe(stats)
            
            # 多特征分析
            st.subheader("多特征分析")
            
            # 特征选择
            default_features = list(data.columns[1:-1])
            selected_features = st.multiselect(
                "选择要分析的特征",
                options=data.columns[1:],
                default=default_features
            )
            
            if len(selected_features) > 0:
                # 方差分析
                st.subheader("特征方差分析")
                variance = data[selected_features].var()
                fig_var = go.Figure()
                fig_var.add_trace(go.Bar(x=variance.index, y=variance.values))
                fig_var.update_layout(title="特征方差分布")
                st.plotly_chart(fig_var)
                
                # 相关性热力图
                st.subheader("特征相关性热力图")
                corr = data[selected_features].corr()
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr,
                    x=selected_features,
                    y=selected_features,
                    colorscale='RdBu'
                ))
                st.plotly_chart(fig_corr)
                
                # PCA分析
                if len(selected_features) >= 3:
                    with st.spinner('正在进行PCA分析...'):
                        st.subheader("PCA 3D可视化")
                        X = data[selected_features]
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        pca = PCA(n_components=3)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        fig_pca = go.Figure(data=[go.Scatter3d(
                            x=X_pca[:, 0],
                            y=X_pca[:, 1],
                            z=X_pca[:, 2],
                            mode='markers',
                            marker=dict(
                                size=5,
                                color=X_pca[:, 0],
                                colorscale='Viridis',
                            )
                        )])
                        st.plotly_chart(fig_pca)
                        
                        # 显示前10个主成分的特征权重
                        n_components = min(10, len(selected_features))
                        pca = PCA(n_components=n_components)
                        pca.fit(X_scaled)
                        
                        components_df = pd.DataFrame(
                            pca.components_,
                            columns=selected_features,
                            index=[f'PC{i+1}' for i in range(n_components)]
                        )
                        
                        st.write(f"前{n_components}个主成分的特征权重：")
                        st.dataframe(components_df.style.background_gradient(cmap='coolwarm'))

            # 在页面最下方添加导航按钮
            st.markdown("---")  # 添加分隔线
            col1, col2, col3, col4, col5 = st.columns(5)
            with col4:
                if st.button("下一步 ➡️"):
                    st.session_state.current_page += 1
                    st.rerun()

        elif st.session_state.current_page == 1:
            # 异常值检测页面
            st.header("异常值检测")
            
            # 选择要检查的列
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols and numeric_cols[0] == data.columns[0]:
                numeric_cols = numeric_cols[1:]  # 如果第一列是数值列，也排除它
            
            outlier_cols = st.multiselect(
                "选择要检查异常值的列",
                options=data.columns[1:],  # 从第二列开始选择
                default=numeric_cols
            )
            
            # 选择异常检测算法
            outlier_method = st.selectbox(
                "选择异常检测算法",
                [
                    "LOF（局部离群因子）",
                    "IForest（孤立森林）",
                    "KNN（K最近邻）",
                    "HBOS（基于直方图的异常检测）",
                    "OCSVM（单类支持向量机）",
                    "ABOD（角度基异常检测）",
                    "CBLOF（基于聚类的局部异常检测）",
                    "COF（连接异常因子）",
                    "LSCP（局部敏感性对比投影）",
                    "PCA（主成分分析）",
                    "SOD（基于子空间的异常检测）",
                    "SOS（基于统计的异常检测）",
                    "VAE（变分自编码器）",
                    "AutoEncoder（自动编码器）",
                    "MO_GAAL（多目标遗传算法异常检测）",
                    "SO_GAAL（单目标遗传算法异常检测）"
                ]
            )
            
            # 异常检测参数设置
            st.header("算法参数设置")
            if outlier_method == "LOF（局部离群因子）":
                n_neighbors = st.slider("邻居数量", 1, 50, 20)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "IForest（孤立森林）":
                n_estimators = st.slider("树的数量", 50, 200, 100)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "KNN（K最近邻）":
                n_neighbors = st.slider("邻居数量", 1, 50, 20)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "HBOS（基于直方图的异常检测）":
                n_bins = st.slider("直方图箱数", 5, 50, 10)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "OCSVM（单类支持向量机）":
                kernel = st.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"])
                nu = st.slider("异常样本比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "ABOD（角度基异常检测）":
                n_neighbors = st.slider("邻居数量", 1, 50, 20)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "CBLOF（基于聚类的局部异常检测）":
                n_clusters = st.slider("聚类数量", 2, 20, 8)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "COF（连接异常因子）":
                n_neighbors = st.slider("邻居数量", 1, 50, 20)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "LSCP（局部敏感性对比投影）":
                n_neighbors = st.slider("邻居数量", 1, 50, 20)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "SOD（基于子空间的异常检测）":
                n_neighbors = st.slider("邻居数量", 1, 50, 20)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "SOS（基于统计的异常检测）":
                perplexity = st.slider("困惑度", 5, 50, 20)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "VAE（变分自编码器）":
                encoding_neurons = st.slider("编码层神经元数量", 8, 128, 32)
                # 创建编码器和解码器的对称结构
                encoder_neurons = [len(outlier_cols), encoding_neurons, encoding_neurons//2]
                decoder_neurons = [encoding_neurons//2, encoding_neurons, len(outlier_cols)]
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                epochs = st.slider("训练轮数", 10, 100, 50)
                batch_size = st.slider("批次大小", 8, 256, 32)
                
            elif outlier_method == "AutoEncoder（自动编码器）":
                encoding_neurons = st.slider("编码层神经元数量", 8, 128, 32)
                # 创建对称的神经网络结构
                hidden_neurons = [encoding_neurons, encoding_neurons//2, encoding_neurons//2, encoding_neurons]
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                epochs = st.slider("训练轮数", 10, 100, 50)
                batch_size = st.slider("批次大小", 8, 256, 32)
                
            elif outlier_method == "MO_GAAL（多目标遗传算法异常检测）":
                k = st.slider("生成器数量", 2, 10, 4)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                
            elif outlier_method == "SO_GAAL（单目标遗传算法异常检测）":
                stop_epochs = st.slider("停止轮数", 10, 100, 50)
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
            
            elif outlier_method == "PCA（主成分分析）":
                contamination = st.slider("异常比例", 0.01, 0.5, 0.1)
                n_components = st.slider("主成分数量", 2, min(10, len(outlier_cols)), min(3, len(outlier_cols)))

            if st.button("检查异常点"):
                with st.spinner('正在进行异常点检测...'):
                    X = data[outlier_cols]
                    
                    # 标准化数据
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    try:
                        # 根据选择的方法进行异常检测
                        if outlier_method == "LOF（局部离群因子）":
                            detector = LOF(n_neighbors=n_neighbors, contamination=contamination)
                        elif outlier_method == "IForest（孤立森林）":
                            detector = IForest(n_estimators=n_estimators, contamination=contamination)
                        elif outlier_method == "KNN（K最近邻）":
                            detector = KNN(n_neighbors=n_neighbors, contamination=contamination)
                        elif outlier_method == "HBOS（基于直方图的异常检测）":
                            detector = HBOS(n_bins=n_bins, contamination=contamination)
                        elif outlier_method == "OCSVM（单类支持向量机）":
                            detector = OCSVM(kernel=kernel, nu=nu)
                        elif outlier_method == "ABOD（角度基异常检测）":
                            detector = ABOD(n_neighbors=n_neighbors, contamination=contamination)
                        elif outlier_method == "CBLOF（基于聚类的局部异常检测）":
                            detector = CBLOF(n_clusters=n_clusters, contamination=contamination)
                        elif outlier_method == "COF（连接异常因子）":
                            detector = COF(n_neighbors=n_neighbors, contamination=contamination)
                        elif outlier_method == "LSCP（局部敏感性对比投影）":
                            detector = LSCP(detector_list=[LOF(), IForest(), HBOS()], contamination=contamination)
                        elif outlier_method == "SOD（基于子空间的异常检测）":
                            detector = SOD(n_neighbors=n_neighbors, contamination=contamination)
                        elif outlier_method == "SOS（基于统计的异常检测）":
                            detector = SOS(perplexity=perplexity, contamination=contamination)
                        elif outlier_method == "VAE（变分自编码器）":
                            detector = VAE(encoder_neurons=encoder_neurons, 
                                        decoder_neurons=decoder_neurons, 
                                        contamination=contamination, 
                                        epochs=epochs, 
                                        batch_size=batch_size)
                        elif outlier_method == "AutoEncoder（自动编码器）":
                            detector = AutoEncoder(hidden_neurons=hidden_neurons, contamination=contamination, epochs=epochs, batch_size=batch_size)
                        elif outlier_method == "MO_GAAL（多目标遗传算法异常检测）":
                            detector = MO_GAAL(k=k, contamination=contamination)
                        elif outlier_method == "SO_GAAL（单目标遗传算法异常检测）":
                            detector = SO_GAAL(stop_epochs=stop_epochs, contamination=contamination)
                        elif outlier_method == "PCA（主成分分析）":
                            from pyod.models.pca import PCA as PyODPCA
                            detector = PyODPCA(contamination=contamination, n_components=n_components)
                        
                        # 训练检测器并获取预测结果
                        outliers = detector.fit_predict(X_scaled)
                        st.session_state.outliers = outliers
                        
                        # 显示异常点
                        outlier_indices = np.where(outliers == 1)[0]
                        st.write(f"发现 {len(outlier_indices)} 个异常点")
                        st.dataframe(data.iloc[outlier_indices])
                        
                        # 在成功检测后设置标志
                        st.session_state.outliers_checked = True
                        
                        # 显示检测结果
                        st.success(f"检测到 {len(outlier_indices)} 个异常值")
                        
                    except Exception as e:
                        st.error(f"异常检测出错：{str(e)}")
                        st.session_state.outliers_checked = False
                
            if st.button("移除异常点") and st.session_state.outliers is not None:
                with st.spinner('正在移除异常点...'):
                    normal_indices = st.session_state.outliers == 0
                    cleaned_data = data[normal_indices].copy()
                    st.session_state.processed_data = cleaned_data
                    st.success(f"已移除 {sum(~normal_indices)} 个异常点")
                    st.download_button(
                        "下载处理后的数据",
                        cleaned_data.to_csv(index=False).encode('utf-8'),
                        "cleaned_data.csv",
                        "text/csv"
                    )

            # 在异常值检测页面底部添加导航按钮
            st.markdown("---")  # 添加分隔线
            col1, col2, col3, col4 = st.columns(4)
            with col2:
                if st.button("⬅️ 上一步", key="prev_2"):
                    st.session_state.current_page = 0
                    st.rerun()
            with col4:
                if st.button("下一步 ➡️", key="next_2"):
                    st.session_state.current_page = 2
                    st.rerun()

        elif st.session_state.current_page == 2:
            # 模型训练页面
            st.header("模型训练")
            
            # 选择目标列和特征列
            target_col = st.selectbox(
                "选择目标列", 
                data.columns[1:],  # 从第二列开始选择
                index=len(data.columns)-2  # 默认选择倒数第二列
            )
            
            available_features = [col for col in data.columns[1:] if col != target_col]  # 从第二列开始，排除目标列
            feature_cols = st.multiselect(
                "选择特征列",
                options=available_features,
                default=available_features
            )
            
            # 选择机器学习算法
            model_type = st.selectbox(
                "选择机器学习算法",
                [
                    "Logistic Regression（逻辑回归）",
                    "K-Nearest Neighbors（KNN）",
                    "Naive Bayes（朴素贝叶斯）",
                    "Decision Tree（决策树）",
                    "Random Forest（随机森林）",
                    "Gradient Boosting Classifier（梯度提升分类器）",
                    "AdaBoost Classifier（AdaBoost分类器）",
                    "SVM（支持向量机）",
                    "XGBoost（XGBoost分类器）",
                    "LightGBM（LightGBM分类器）"
                ]
            )
            
            # 添加数据集划分比例设置
            st.subheader("数据集划分设置")
            test_size = st.slider(
                "测试集比例",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="设置测试集占总数据集的比例，范围0.1-0.5"
            )
            
            # 模型参数设置
            st.subheader("模型参数设置")
            if model_type == "Logistic Regression（逻辑回归）":
                C = st.slider("正则化强度", 0.01, 10.0, 1.0)
                max_iter = st.slider("最大迭代次数", 100, 1000, 100)
                
            elif model_type == "K-Nearest Neighbors（KNN）":
                n_neighbors = st.slider("邻居数量", 1, 20, 5)
                
            elif model_type == "Naive Bayes（朴素贝叶斯）":
                # No additional parameters needed
                pass
                
            elif model_type == "Decision Tree（决策树）":
                max_depth = st.slider("最大深度", 1, 20, 5)
                min_samples_split = st.slider("最小分割样本数", 2, 20, 5)
                min_samples_leaf = st.slider("最小叶节点样本数", 1, 20, 5)
                max_features = st.slider("最大特征数", 1, len(feature_cols), len(feature_cols))
                
            elif model_type == "Random Forest（随机森林）":
                n_estimators = st.slider("树的数量", 100, 500, 100)
                max_depth = st.slider("最大深度", 1, 20, 5)
                min_samples_split = st.slider("最小分割样本数", 2, 20, 5)
                min_samples_leaf = st.slider("最小叶节点样本数", 1, 20, 5)
                max_features = st.slider("最大特征数", 1, len(feature_cols), len(feature_cols))
                
            elif model_type == "Gradient Boosting Classifier（梯度提升分类器）":
                n_estimators = st.slider("树的数量", 100, 500, 100)
                learning_rate = st.slider("学习率", 0.01, 0.5, 0.1)
                max_depth = st.slider("最大深度", 1, 20, 5)
                
            elif model_type == "AdaBoost Classifier（AdaBoost分类器）":
                n_estimators = st.slider("树的数量", 100, 500, 100)
                learning_rate = st.slider("学习率", 0.01, 0.5, 0.1)
                
            elif model_type == "SVM（支持向量机）":
                kernel = st.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"])
                C = st.slider("正则化强度", 0.01, 10.0, 1.0)
                gamma = st.selectbox("Gamma", ["scale", "auto"])
                
            elif model_type == "XGBoost（XGBoost分类器）":
                n_estimators = st.slider("树的数量", 100, 500, 100)
                learning_rate = st.slider("学习率", 0.01, 0.5, 0.1)
                max_depth = st.slider("最大深度", 1, 20, 5)
                
            elif model_type == "LightGBM（LightGBM分类器）":
                n_estimators = st.slider("树的数量", 100, 500, 100)
                learning_rate = st.slider("学习率", 0.01, 0.5, 0.1)
                max_depth = st.slider("最大深度", 1, 20, 5)

            if st.button("开始训练"):
                with st.spinner('正在训练模型...'):
                    # 准备数据
                    X = data[feature_cols]
                    y = data[target_col]
                    
                    # 划分训练集和测试集
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # 根据选择的模型类型创建模型
                    if model_type == "Logistic Regression（逻辑回归）":
                        model = LogisticRegression(C=C, max_iter=max_iter)
                    elif model_type == "K-Nearest Neighbors（KNN）":
                        model = KNeighborsClassifier(n_neighbors=n_neighbors)
                    elif model_type == "Naive Bayes（朴素贝叶斯）":
                        model = GaussianNB()
                    elif model_type == "Decision Tree（决策树）":
                        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
                    elif model_type == "Random Forest（随机森林）":
                        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)
                    elif model_type == "Gradient Boosting Classifier（梯度提升分类器）":
                        model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                    elif model_type == "AdaBoost Classifier（AdaBoost分类器）":
                        model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
                    elif model_type == "SVM（支持向量机）":
                        model = SVC(kernel=kernel, C=C, gamma=gamma)
                    elif model_type == "XGBoost（XGBoost分类器）":
                        model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                    elif model_type == "LightGBM（LightGBM分类器）":
                        model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
                    
                    # 训练模型
                    model.fit(X_train, y_train)
                    
                    # 预测
                    y_pred = model.predict(X_test)
                    
                    # 显示混淆矩阵
                    st.subheader("混淆矩阵")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # 创建混淆矩阵的DataFrame
                    unique_labels = sorted(list(set(y_test) | set(y_pred)))
                    cm_df = pd.DataFrame(
                        cm,
                        index=[f'实际: {label}' for label in unique_labels],
                        columns=[f'预测: {label}' for label in unique_labels]
                    )
                    
                    # 显示混淆矩阵表格
                    st.dataframe(
                        cm_df.style.background_gradient(cmap='Blues', axis=None)
                        .format("{:.0f}")
                    )
                    
                    # 显示分类报告
                    st.subheader("分类报告")
                    report = classification_report(y_test, y_pred)
                    st.text(report)

            # 在页面最下方添加导航按钮
            st.markdown("---")  # 添加分隔线
            col1, col2, col3, col4, col5 = st.columns(5)
            with col2:
                if st.button("⬅️ 上一步"):
                    st.session_state.current_page -= 1
                    st.rerun()

    except Exception as e:
        st.error(f"发生错误：{str(e)}")
else:
    st.info("请先上传数据文件") 