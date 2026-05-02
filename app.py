import streamlit as st
import pandas as pd
import numpy as np
import dill
import pickle
import plotly.express as px
import plotly.graph_objects as go
from lifetimes import BetaGeoFitter, GammaGammaFitter

# ─── Page Config ───
st.set_page_config(
    page_title="Customer LTV & Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ─── Load Models ───
@st.cache_resource
def load_models():
    bgf = dill.load(open('bgf_model.pkl', 'rb'))
    ggf = dill.load(open('ggf_model.pkl', 'rb'))
    xgb_model = pickle.load(open('xgb_churn_model.pkl', 'rb'))
    return bgf, ggf, xgb_model

@st.cache_data
def load_default_data():
    return pd.read_csv('rfm_with_predictions.csv', index_col=0)

bgf, ggf, xgb_model = load_models()

# ─── Header ───
st.title("📊 Customer LTV & Churn Intelligence")
st.markdown("Predict lifetime value, identify churn risk, and simulate retention strategies.")
st.divider()

# ─── Sidebar: Data Source ───
st.sidebar.header("Data Source")
data_mode = st.sidebar.radio("Choose input:", ["Use existing predictions", "Upload new transactions"])

if data_mode == "Upload new transactions":
    st.sidebar.markdown("**Required columns:** `customer_id`, `order_date`, `total_amount_usd`")
    uploaded_orders = st.sidebar.file_uploader("Orders CSV", type="csv")
    uploaded_customers = st.sidebar.file_uploader("Customers CSV", type="csv")

    if uploaded_orders and uploaded_customers:
        from lifetimes.utils import summary_data_from_transaction_data

        orders_new = pd.read_csv(uploaded_orders)
        customers_new = pd.read_csv(uploaded_customers)
        orders_new['order_date'] = pd.to_datetime(orders_new['order_date'])

        rfm = summary_data_from_transaction_data(
            orders_new, 'customer_id', 'order_date',
            monetary_value_col='total_amount_usd'
        )
        rfm = rfm[rfm['frequency'] > 0]

        rfm['prob_alive'] = bgf.conditional_probability_alive(
            rfm['frequency'], rfm['recency'], rfm['T']
        )
        rfm['predicted_clv'] = ggf.customer_lifetime_value(
            bgf, rfm['frequency'], rfm['recency'], rfm['T'],
            rfm['monetary_value'], time=12, discount_rate=0.01
        )

        customers_ml = customers_new.set_index('customer_id')
        customers_ml['tier_encoded'] = customers_ml['membership_tier'].map(
            {'Free': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}
        )
        customers_ml['channel_encoded'] = customers_ml['acquisition_channel'].astype('category').cat.codes

        feature_cols = ['total_orders', 'total_spend_usd', 'avg_order_value_usd',
                        'days_since_last_purchase', 'reviews_given', 'avg_review_score',
                        'returns_made', 'wishlist_items', 'newsletter_subscribed',
                        'tier_encoded', 'channel_encoded']

        X_new = customers_ml.reindex(rfm.index)[feature_cols].fillna(0)
        rfm['churn_probability'] = xgb_model.predict_proba(X_new)[:, 1]

        rfm['value_tier'] = pd.qcut(rfm['predicted_clv'], q=3, labels=['Low', 'Medium', 'High'])
        rfm['churn_risk'] = pd.qcut(rfm['churn_probability'], q=3, labels=['Low', 'Medium', 'High'])

        data = rfm
        st.sidebar.success(f"Processed {len(data):,} customers")
    else:
        st.info("Upload both CSVs to get started, or switch to existing predictions.")
        st.stop()
else:
    data = load_default_data()

# ─── KPI Row ───
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers", f"{len(data):,}")
col2.metric("Median CLV (12mo)", f"${data['predicted_clv'].median():,.0f}")
col3.metric("High Churn Risk", f"{(data['churn_risk'] == 'High').sum():,}")
col4.metric(
    "Revenue at Risk",
    f"${data[data['churn_risk'] == 'High']['predicted_clv'].sum():,.0f}"
)

st.divider()

# ─── Tab Layout ───
tab1, tab2, tab3 = st.tabs(["🔥 Risk Heatmap", "🔍 Customer Lookup", "🎯 What-If Simulator"])

# ═══════════════════════════════════════
# TAB 1: Risk Heatmap
# ═══════════════════════════════════════
with tab1:
    st.subheader("Churn Risk vs Customer Value")
    st.markdown("**Top-right = danger zone.** High-value customers likely to leave.")

    heatmap_data = pd.crosstab(
        data['churn_risk'], data['value_tier']
    ).reindex(index=['High', 'Medium', 'Low'], columns=['Low', 'Medium', 'High'])

    fig_heat = px.imshow(
        heatmap_data.values,
        x=['Low Value', 'Medium Value', 'High Value'],
        y=['High Risk', 'Medium Risk', 'Low Risk'],
        text_auto=True,
        color_continuous_scale='RdYlGn_r',
        labels=dict(color="Customers"),
        aspect="auto"
    )
    fig_heat.update_layout(height=400, font=dict(size=14))
    st.plotly_chart(fig_heat, width='stretch')

    danger = data[(data['value_tier'] == 'High') & (data['churn_risk'] == 'High')]
    st.error(f"⚠️ **{len(danger)} high-value customers at high churn risk** — ${danger['predicted_clv'].sum():,.0f} in projected revenue at stake")

    with st.expander("View High-Risk / High-Value Customers"):
        display_cols = ['predicted_clv', 'churn_probability', 'frequency', 'recency', 'monetary_value']
        available_cols = [c for c in display_cols if c in danger.columns]
        st.dataframe(
            danger[available_cols].sort_values('predicted_clv', ascending=False).head(20),
            width='stretch'
        )

# ═══════════════════════════════════════
# TAB 2: Customer Lookup
# ═══════════════════════════════════════
with tab2:
    st.subheader("Individual Customer Profile")

    customer_ids = data.index.tolist()
    selected_id = st.selectbox("Select Customer ID", customer_ids)

    if selected_id:
        cust = data.loc[selected_id]

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted 12-Month CLV", f"${cust['predicted_clv']:.2f}")
        col2.metric("Churn Probability", f"{cust.get('churn_probability', 0):.1%}")
        col3.metric("P(Alive) — BG/NBD", f"{cust.get('prob_alive', 0):.1%}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Total Transactions", f"{int(cust['frequency'])}")
        col5.metric("Recency (days)", f"{int(cust['recency'])}")
        col6.metric("Avg Transaction Value", f"${cust['monetary_value']:.2f}")

        risk = cust.get('churn_risk', 'Unknown')
        value = cust.get('value_tier', 'Unknown')
        if risk == 'High' and value == 'High':
            st.error("🚨 PRIORITY: High-value customer at high churn risk")
        elif risk == 'High':
            st.warning("⚠️ Elevated churn risk — consider retention outreach")
        else:
            st.success("✅ Customer appears healthy")

# ═══════════════════════════════════════
# TAB 3: What-If Simulator
# ═══════════════════════════════════════
with tab3:
    st.subheader("Retention Strategy Simulator")
    st.markdown("Simulate how a discount changes a customer's expected future behavior.")

    sim_id = st.selectbox("Select Customer", customer_ids, key="sim_customer")
    cust_sim = data.loc[sim_id]

    col_left, col_right = st.columns([1, 2])

    with col_left:
        discount = st.slider("Discount Offered (%)", 0, 50, 10, step=5)
        st.caption("Assumption: Each 5% discount increases expected purchases by ~8% and retention probability by ~3%")

        boost_factor = 1 + (discount / 5) * 0.08
        retention_boost = min((discount / 5) * 0.03, 0.25)

        base_purchases = float(np.atleast_1d(bgf.conditional_expected_number_of_purchases_up_to_time(
            365, cust_sim['frequency'], cust_sim['recency'], cust_sim['T']
        )).flatten()[0])
        boosted_purchases = base_purchases * boost_factor

        base_alive = float(np.atleast_1d(bgf.conditional_probability_alive(
            cust_sim['frequency'], cust_sim['recency'], cust_sim['T']
        )).flatten()[0])
        boosted_alive = min(base_alive + retention_boost, 0.99)

        base_revenue = float(base_purchases * cust_sim['monetary_value'])
        discount_cost = float(boosted_purchases * cust_sim['monetary_value'] * (discount / 100))
        boosted_revenue = float(boosted_purchases * cust_sim['monetary_value']) - discount_cost

        net_gain = boosted_revenue - base_revenue

        
        base_purchases = float(base_purchases)
        boosted_purchases = float(boosted_purchases)
        base_alive = float(base_alive)
        boosted_alive = float(boosted_alive)

    with col_right:
        metrics = ['Expected Purchases (12mo)', 'P(Alive)', 'Expected Revenue ($)']
        base_vals = [base_purchases, base_alive, base_revenue]
        boosted_vals = [boosted_purchases, boosted_alive, boosted_revenue]

        fig_compare = go.Figure()
        fig_compare.add_trace(go.Bar(
            name='No Discount', x=metrics, y=base_vals,
            marker_color='#EF5350', text=[f"{v:.2f}" for v in base_vals], textposition='outside'
        ))
        fig_compare.add_trace(go.Bar(
            name=f'{discount}% Discount', x=metrics, y=boosted_vals,
            marker_color='#66BB6A', text=[f"{v:.2f}" for v in boosted_vals], textposition='outside'
        ))
        fig_compare.update_layout(barmode='group', height=400, title="Impact Simulation")
        st.plotly_chart(fig_compare, width='stretch')

    st.divider()
    roi_col1, roi_col2, roi_col3 = st.columns(3)
    roi_col1.metric("Discount Cost", f"${discount_cost:.2f}")
    roi_col2.metric("Revenue Lift", f"${boosted_revenue - base_revenue + discount_cost:.2f}")
    roi_col3.metric(
        "Net ROI",
        f"${net_gain:.2f}",
        delta=f"{'Profitable' if net_gain > 0 else 'Not worth it'}",
        delta_color="normal" if net_gain > 0 else "inverse"
    )

# ─── Footer ───
st.divider()
st.caption("Built with BG/NBD + Gamma-Gamma (lifetimes) for CLV • XGBoost for Churn • Streamlit for UI")
