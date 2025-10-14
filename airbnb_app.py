import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import folium
from streamlit_folium import st_folium
import requests
from geopy.distance import great_circle
from shapely.geometry import Point, shape


st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        width: 300px !important; /* 원하는 너비로 조절하세요 */
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# --- 0. 페이지 기본 설정 ---
st.set_page_config(page_title="뉴욕 에어비앤비 가격 예측")



# --- 1. 데이터 및 모델 로딩 ---

# @st.cache_resource 데코레이터는 무거운 데이터를 한번만 로드하게 해 앱 속도를 높여줍니다.
@st.cache_resource
# model_all_in_one.pkl' 파일 하나만 불러와서 모든 정보를 준비하는 함수
def load_all_data():
    # 1. '올인원' 파일 하나만 불러옵니다.
    with open("model_all_in_one.pkl", 'rb') as f:
        data = joblib.load(f)
        
    # 2. 딕셔너리에서 필요한 모든 변수를 꺼냅니다.
    model = data["model"]
    X_columns = data["X_columns"]
    std_residual = data["std_residual"]
    amenities_keywords = data["amenities_keywords"]
    categorical_values = data["categorical_values"]
    
    # 3. 거리 계산에 필요한 외부 데이터 파일들도 함께 불러옵니다.
    bus_df = pd.read_csv("bus.csv")
    subway_df = pd.read_csv("subway.csv")
    airport_df = pd.read_csv("airport.csv")
    
    with open("nyc_neighborhoods.geojson", "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    
    # 4. 준비된 모든 변수를 밖으로 전달합니다.
    return model, X_columns, std_residual, amenities_keywords, categorical_values, bus_df, subway_df, airport_df, geojson_data

model, X_columns, std_residual, amenities_keywords, categorical_values, bus_df, subway_df, airport_df, geojson_data = load_all_data()



# --- 2. 핵심 기능 함수 정의 ---

# GeoJSON 데이터를 기반으로 위도/경도에 해당하는 지역 이름을 반환
def find_region_from_coords(lon, lat, geojson):
    clicked_point = Point(lon, lat)
    for feature in geojson['features']:
        polygon = shape(feature['geometry'])
        if polygon.contains(clicked_point):
            borough = feature['properties'].get('BoroName', 'N/A')
            neighborhood = feature['properties'].get('NTAName', 'N/A')
            return borough, neighborhood
    return None, None

# 주어진 좌표에서 가장 가까운 지점의 좌표를 찾는 함수
def find_nearest_point(target_lat, target_lon, points_df, lat_col, lon_col):
    distances = [great_circle((target_lat, target_lon), (row_lat, row_lon)).meters for row_lat, row_lon in zip(points_df[lat_col], points_df[lon_col])]
    nearest_index = np.argmin(distances)
    nearest_coords = (points_df.iloc[nearest_index][lon_col], points_df.iloc[nearest_index][lat_col])
    return nearest_coords

# Mapbox Directions API를 호출하여 실제 경로 거리를 계산하는 함수
def get_mapbox_distance(start_lon, start_lat, end_lon, end_lat, profile='walking'):
    api_key = st.secrets["MAPBOX_TOKEN"]
    url = (f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{start_lon},{start_lat};{end_lon},{end_lat}?access_token={api_key}")
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if response.status_code == 200 and 'routes' in data and data['routes']:
            return round(data['routes'][0]['distance'])
        else: return None
    except requests.exceptions.RequestException: return None



# --- 3. 세션 상태 초기화 ---

if 'map_results' not in st.session_state:
    st.session_state.map_results = None



# --- 4. 사이드바 UI 구성 ---

with st.sidebar:
    st.title("⚙️ 숙소 조건 설정")
    st.markdown("---")

    room_type = st.selectbox("룸 타입 (Room Type)", sorted(categorical_values['room_type']))
    accommodates = st.slider("수용 인원 (Accommodates)", 1, 16, 2)
    bedrooms = st.slider("침실 수 (Bedrooms)", 0, 10, 1)
    bathrooms = st.slider("욕실 수 (Bathrooms)", 0.0, 8.0, 1.0, 0.5)
    beds = st.slider("침대 수 (Beds)", 1, 20, 1)
    st.markdown("---")
    minimum_nights = st.number_input('최소 숙박일 (Minimum Nights)', 1, value=1)
    maximum_nights = st.number_input('최대 숙박일 (Maximum Nights)', 1, value=1125, help="모델 학습 시 최대값(1125)을 기준으로 설정했습니다.")
    st.markdown("---")
    review_scores_rating = st.slider("리뷰 평점 (Review Score Rating)", 1.0, 5.0, 4.5, 0.01)
    instant_bookable = st.radio('즉시 예약 가능 (Instant Bookable)', ('Yes', 'No')) == 'Yes'
    st.markdown("---")
    amenities = st.multiselect('편의시설 (Amenities)', options=sorted(amenities_keywords))
    amenity_categories = " ".join(amenities)
    st.markdown("---")
    st.subheader("🗺️ 위치 선택")
    st.markdown("지도에서 원하는 숙소 위치를 클릭하세요.")

    borough_colors = pd.DataFrame({'BoroName': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'], 'color_value': [1,2,3,4,5]})
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10, tiles="cartodbpositron")
    folium.Choropleth(geo_data=geojson_data, data=borough_colors, columns=['BoroName', 'color_value'], key_on='feature.properties.BoroName', fill_color='Set3', fill_opacity=0.5, line_opacity=0.2).add_to(m)
    map_data = st_folium(m, height=200, width=200)

    if map_data and map_data['last_clicked']:
        lat, lon = map_data['last_clicked']['lat'], map_data['last_clicked']['lng']
        with st.spinner("위치 정보 계산 중..."):
            borough, neighborhood = find_region_from_coords(lon, lat, geojson_data)
            sub_coords, bus_coords, air_coords = find_nearest_point(lat, lon, subway_df, 'stop_lat', 'stop_lon'), find_nearest_point(lat, lon, bus_df, 'stop_lat', 'stop_lon'), find_nearest_point(lat, lon, airport_df, 'latitude', 'longitude')
            sub_dist, bus_dist, air_dist = get_mapbox_distance(sub_coords[0], sub_coords[1], lon, lat, 'walking'), get_mapbox_distance(bus_coords[0], bus_coords[1], lon, lat, 'walking'), get_mapbox_distance(air_coords[0], air_coords[1], lon, lat, 'driving')
            st.session_state.map_results = {'latitude': lat, 'longitude': lon, 'walk_subway(m)': sub_dist, 'walk_bus(m)': bus_dist, 'car_airport(m)': air_dist, 'neighbourhood_group_cleansed': borough, 'neighbourhood_cleansed': neighborhood}
    
    # 계산된 위치 정보를 사이드바에 표시
    if st.session_state.map_results:
        results = st.session_state.map_results

        st.subheader("📍 자동 입력된 위치 정보")

        st.markdown(f"""
        **자치구:** {results.get('neighbourhood_group_cleansed', '정보 없음')}  
        **동네:** {results.get('neighbourhood_cleansed', '정보 없음')}  
        ---
        🚇 **지하철:** {results.get('walk_subway(m)', '계산 실패')} m  
        🚌 **버스:** {results.get('walk_bus(m)', '계산 실패')} m  
        ✈️ **공항:** {results.get('car_airport(m)', '계산 실패')} m
        """)



# --- 5. 메인 화면 UI 구성 ---

st.image("airbnb_logo.png", width=150)
st.markdown("""
##### 💰 숙소 가격 예측 결과 확인
""")
st.markdown("사이드 바에서 모든 변수 설정 완료 후 가격 예측을 진행해주세요")

predict_button = st.button("가격 예측 실행하기 ✔️")

if predict_button:
    if not st.session_state.map_results or not st.session_state.map_results.get('neighbourhood_cleansed'):
        st.error("🚨 사이드바의 지도에서 유효한 지역(동네)을 먼저 선택해주세요!")
    else:
        with st.spinner("AI가 최적의 가격을 예측하고 있습니다..."):
            
            input_data = st.session_state.map_results.copy()
            input_data.update({
                'room_type': room_type, 'accommodates': accommodates, 'bedrooms': bedrooms,
                'bathrooms': bathrooms, 'beds': beds, 'minimum_nights': minimum_nights,
                'maximum_nights': maximum_nights, 'review_scores_rating': review_scores_rating,
                'instant_bookable': instant_bookable, 'amenity_categories': amenity_categories,
            })
            input_df = pd.DataFrame([input_data])
            for col in X_columns:
                if col not in input_df.columns:
                    input_df[col] = "" if col == 'amenity_categories' else 0
            input_df = input_df[X_columns]

            try:
                log_price = model.predict(input_df)[0]
                pred_price = np.expm1(log_price)

                lower = np.maximum(0, pred_price - 0.5 * std_residual)
                upper = pred_price + 0.5 * std_residual
                
                st.success(f"🏷️ 예측 숙소 가격: **${pred_price:,.2f}**")
                st.info(f"📏 예측 오차 범위: **${lower:,.2f} ~ {upper:,.2f}**")

            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {e}")
                st.dataframe(input_df)


st.markdown("""
--- 
##### 🧠 예측 모델 정보

""")
with st.expander("🔎 어떤 항목들이 예측에 사용되나요?"):
    st.markdown("""
    - 숙소 유형 (Entire home, Private room 등)
    - 위치 (위도/경도)
    - 숙박 가능 인원 수
    - 욕실, 침실 수
    - 최소 / 최대 숙박일
    - 즉시 예약 가능 여부
    - 리뷰 평점
    - 제공 어메니티 수
    """)

with st.expander("📊 모델 정보"):
    st.markdown("""
    - 알고리즘: CatBoost Regressor
    - 평가 지표: RMSE = 53.7387, MAE = 36.8443, R2 = 0.7145
    - 로그 변환(y) + 오차범위 표시 기능 포함
    """)

st.markdown("""
---
##### 📌 프로젝트 정보

👨‍💻 프로젝트명 | 2025 내일배움캠프 심화 프로젝트  
👥 팀 멤버 | 김윤환, 김소영, 조현도, 임은지  
📁 데이터 출처 | Airbnb (크롤링 데이터 기반)  
📬 문의 메일 | `8조_세계정복김순대@gmail.com`
""")