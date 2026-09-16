import React, { useState, useMemo } from 'react';
import allData from './data/mockdata.json';
import { 
  Home, FileText, AlertTriangle, BarChart2, PieChart as PieIcon, 
  Users, Settings, Bell, IndianRupee, CheckCircle2, 
  X, BrainCircuit, ShieldAlert, Activity, Search, MapPin, Download, Calendar, Menu, FileCheck, RefreshCw
} from 'lucide-react';
import { LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, BarChart, Bar, Legend } from 'recharts';

export default function App() {
  const allConstituencies = Object.keys(allData);
  const [selectedConstituency, setSelectedConstituency] = useState(allConstituencies[0]);
  const [selectedProjectDetails, setSelectedProjectDetails] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [workOrderSearch, setWorkOrderSearch] = useState('');

  const currentData = allData[selectedConstituency];
  const { summary, trend_data, projects } = currentData;

  const calculatedUtilization = useMemo(() => {
    if (summary.utilization) return summary.utilization;
    if (trend_data && trend_data.length > 0) {
      const latest = trend_data[trend_data.length - 1];
      if (latest.sanctioned > 0) {
        return ((latest.expenditure / latest.sanctioned) * 100).toFixed(1);
      }
    }
    const charSum = selectedConstituency.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
    return (62 + (charSum % 28)).toFixed(1);
  }, [summary, trend_data, selectedConstituency]);

  const risk_distribution = [
    { name: "High Risk", value: summary.high_risk_orders, fill: "#EF4444" },
    { name: "Medium Risk", value: Math.floor(summary.total_work_orders * 0.18), fill: "#F97316" },
    { name: "Low Risk", value: Math.floor(summary.total_work_orders * 0.32), fill: "#FACC15" },
    { name: "Minimal Risk", value: Math.floor(summary.total_work_orders * 0.45), fill: "#10B981" }
  ];

  const risk_trend = [
    { month: "Jan '25", high: 12, medium: 6, low: 2 }, { month: "Feb '25", high: 16, medium: 8, low: 3 },
    { month: "Mar '25", high: 12, medium: 7, low: 2 }, { month: "Apr '25", high: 14, medium: 9, low: 4 },
    { month: "May '25", high: summary.high_risk_orders, medium: 10, low: 4 }
  ];

  const schemeData = [
    { category: 'Roads & Infra', allocated: 45, utilized: 38 },
    { category: 'Education', allocated: 25, utilized: 22 },
    { category: 'Water Supply', allocated: 18, utilized: 12 },
    { category: 'Healthcare', allocated: 12, utilized: 10 },
  ];

  const filteredProjects = projects.filter(p => 
    p.work_id.toLowerCase().includes(workOrderSearch.toLowerCase()) || 
    p.description.toLowerCase().includes(workOrderSearch.toLowerCase())
  );

  const avgRisk = useMemo(() => {
    const total = projects.reduce((acc, p) => acc + (p.overall || 0.5), 0);
    return (total / projects.length).toFixed(2);
  }, [projects]);

  const radius = 68;
  const circumference = Math.PI * radius;
  const strokeOffset = circumference - (avgRisk * circumference);
  const gaugeColor = avgRisk > 0.7 ? '#EF4444' : avgRisk > 0.4 ? '#F97316' : '#10B981';
  const riskLabel = avgRisk > 0.7 ? 'HIGH RISK' : avgRisk > 0.4 ? 'MEDIUM RISK' : 'LOW RISK';

  const handleDownloadPDF = () => {
    const reportContent = `
==================================================
EXPERIMENTALISTS - MPLADS AI AUDIT REPORT
==================================================
Topic: Development of an AI-powered system to detect anomalies, fraud, and inefficiencies in MPLAD Scheme implementation
Constituency: ${selectedConstituency}
Member of Parliament: ${summary.mp_name}
Total Funds Allocated: ₹ ${summary.total_funds} Cr
Total Work Orders: ${summary.total_work_orders}
Completed Projects: ${summary.completed_projects} (${summary.completed_pct}%)
Fund Utilization: ${calculatedUtilization}%
High Risk Flags: ${summary.high_risk_orders}
Average Risk Index: ${avgRisk} (${riskLabel})

TOP FLAGGED WORK ORDERS & SHAP TELEMETRY:
${projects.map(p => `- [${p.work_id}] ${p.description} | Amount: ₹ ${p.amount} | Risk: ${p.risk} | Status: ${p.status}`).join('\n')}

Generated securely by Experimentalists Audit Grid Engine.
Date: ${new Date().toLocaleDateString()}
==================================================
    `;
    const blob = new Blob([reportContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `MPLADS_Audit_Report_${selectedConstituency.replace(/[^a-zA-Z0-9]/g, '_')}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const getSubScores = (proj) => {
    return {
      dup: proj.duplication !== undefined ? proj.duplication : Math.min(proj.overall * 1.1, 0.99).toFixed(2),
      del: proj.delay !== undefined ? proj.delay : (proj.overall * 0.9).toFixed(2),
      comp: proj.compliance !== undefined ? proj.compliance : (proj.overall * 0.85).toFixed(2),
      progress: proj.status === 'Completed' ? 100 : proj.status === 'Delayed' ? 35 : 65,
      recDate: `12 Oct 2024`
    };
  };

  return (
    <div>
      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
          * { box-sizing: border-box; font-family: 'Inter', sans-serif; }
          body { margin: 0; padding: 0; }
          ::-webkit-scrollbar { width: 6px; height: 6px; }
          ::-webkit-scrollbar-track { background: transparent; }
          ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 10px; }
        `}
      </style>

      <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', color: '#0F172A', fontSize: '15px' }}>
        
        {/* ================= SIDEBAR (245px width, Large Logo) ================= */}
        <div style={{ width: '245px', backgroundColor: '#FFFFFF', display: 'flex', flexDirection: 'column', justifyContent: 'space-between', zIndex: 30, borderRight: '1px solid #E2E8F0', boxShadow: '2px 0 10px rgba(0,0,0,0.02)', flexShrink: 0 }}>
          <div>
            {/* Expanded Logo: 200px wide × 90px high */}
            <div style={{ padding: '22px 10px', display: 'flex', flexDirection: 'column', alignItems: 'center', borderBottom: '1px solid #F1F5F9' }}>
              <div style={{ width: '200px', height: '90px', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden' }}>
                <img src="/logo.png" alt="Experimentalists Logo" style={{ width: '200px', height: '90px', objectFit: 'contain' }} onError={(e) => { e.target.style.display = 'none'; }} />
              </div>
            </div>
            
            <div style={{ padding: '18px 14px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              {[
                { id: 'overview', label: 'Dashboard', icon: Home },
                { id: 'workorders', label: 'Work Orders', icon: FileText },
                { id: 'riskalerts', label: 'Risk Alerts', icon: AlertTriangle, badge: summary.high_risk_orders, badgeColor: '#EF4444' },
                { id: 'constituency', label: 'Constituency Analysis', icon: BarChart2 },
                { id: 'scheme', label: 'Scheme Analytics', icon: PieIcon },
                { id: 'reports', label: 'Reports', icon: FileText },
                { id: 'audittrail', label: 'Audit Trail', icon: Activity },
                { id: 'users', label: 'User Management', icon: Users },
                { id: 'settings', label: 'Settings', icon: Settings },
              ].map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <div 
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    style={{ 
                      padding: '10px 14px', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                      backgroundColor: isActive ? '#2563EB' : 'transparent',
                      color: isActive ? '#FFFFFF' : '#334155',
                      borderRadius: '8px',
                      cursor: 'pointer', fontSize: '15px', fontWeight: isActive ? 600 : 500,
                      boxShadow: isActive ? '0 4px 12px rgba(37, 99, 235, 0.25)' : 'none',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <Icon size={21} color={isActive ? '#FFFFFF' : '#64748B'} /> {item.label}
                    </div>
                    {item.badge && (
                      <span style={{ backgroundColor: isActive ? '#FFFFFF' : '#EF4444', color: isActive ? '#EF4444' : '#fff', fontSize: '12px', padding: '2px 7px', borderRadius: '10px', fontWeight: 600 }}>
                        {item.badge}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div style={{ padding: '18px', borderTop: '1px solid #F1F5F9', backgroundColor: '#FFFFFF' }}>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
              <div style={{ width: '38px', height: '38px', backgroundColor: '#EFF6FF', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#2563EB', fontWeight: 600, fontSize: '14px' }}>
                AK
              </div>
              <div>
                <div style={{ fontSize: '14px', fontWeight: 600, color: '#0F172A' }}>Arun Kumar</div>
                <div style={{ fontSize: '12px', color: '#64748B', fontWeight: 400 }}>District Authority</div>
              </div>
            </div>
          </div>
        </div>

        {/* ================= MAIN CONTENT ================= */}
        <div style={{ 
          flex: 1, 
          overflowY: 'auto', 
          position: 'relative',
          backgroundImage: "url('/background.png')", 
          backgroundColor: '#F8FAFC',
          backgroundSize: 'cover', 
          backgroundPosition: 'center top',
          backgroundAttachment: 'fixed',
          backgroundRepeat: 'no-repeat'
        }}>
          
          {/* Top Header Bar with prominent SIH 2026 Topic Focus */}
          <div style={{ padding: '18px 32px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', backdropFilter: 'blur(4px)', backgroundColor: 'rgba(255, 255, 255, 0.9)', borderBottom: '1px solid rgba(226, 232, 240, 0.8)' }}>
            <div style={{ display: 'flex', gap: '16px', alignItems: 'center', flex: '1 1 500px' }}>
              <Menu size={24} color="#0F172A" style={{ cursor: 'pointer', flexShrink: 0 }} />
              <div>
                <div style={{ fontSize: '12px', fontWeight: 700, color: '#2563EB', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '2px' }}>SIH 2026 Project Focus</div>
                <div style={{ fontSize: '15px', fontWeight: 700, color: '#0F172A', lineHeight: '1.4' }}>
                  Development of an AI-powered system to detect anomalies, fraud, and inefficiencies in MPLAD Scheme implementation
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexWrap: 'wrap' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: 'rgba(255,255,255,0.95)', padding: '0 12px', height: '44px', borderRadius: '8px', border: '1px solid #E2E8F0', fontSize: '14px', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
                <MapPin size={17} color="#2563EB" />
                <select 
                  value={selectedConstituency} 
                  onChange={(e) => setSelectedConstituency(e.target.value)}
                  style={{ border: 'none', background: 'transparent', outline: 'none', fontWeight: 600, fontSize: '14px', color: '#0F172A', cursor: 'pointer' }}
                >
                  {allConstituencies.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', backgroundColor: 'rgba(255,255,255,0.95)', padding: '0 12px', height: '44px', borderRadius: '8px', border: '1px solid #E2E8F0', fontSize: '14px', color: '#0F172A', fontWeight: 500, boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
                <Calendar size={17} color="#2563EB" /> Sep 2026
              </div>

              <button 
                onClick={handleDownloadPDF}
                style={{ backgroundColor: '#2563EB', color: '#fff', border: 'none', padding: '0 18px', height: '42px', borderRadius: '8px', fontSize: '14px', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', boxShadow: '0 2px 6px rgba(37,99,235,0.2)' }}
              >
                <Download size={16} /> Download Report
              </button>

              <div style={{ position: 'relative', cursor: 'pointer', padding: '10px', backgroundColor: 'rgba(255,255,255,0.95)', borderRadius: '50%', border: '1px solid #E2E8F0', boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }} onClick={() => setActiveTab('riskalerts')}>
                <Bell size={20} color="#0F172A" />
                <span style={{ position: 'absolute', top: 1, right: 1, backgroundColor: '#EF4444', width: '10px', height: '10px', borderRadius: '50%', border: '2px solid #FFFFFF' }}></span>
              </div>
            </div>
          </div>

          <div style={{ padding: '28px 32px', display: 'flex', flexDirection: 'column', gap: '22px', zIndex: 10, position: 'relative' }}>
            
            {/* ================= WELCOME BANNER ================= */}
            <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', padding: '22px 32px', boxShadow: '0 2px 6px rgba(0,0,0,0.02)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '14px' }}>
              <div>
                <div style={{ fontSize: '12px', fontWeight: 700, color: '#2563EB', textTransform: 'uppercase', letterSpacing: '0.8px', marginBottom: '3px' }}>Active Surveillance Command Centre</div>
                <h2 style={{ fontSize: '28px', fontWeight: 800, color: '#1E3A8A', margin: '0 0 4px 0', letterSpacing: '-0.3px' }}>
                  Welcome to {selectedConstituency}
                </h2>
                <p style={{ fontSize: '15px', color: '#475569', margin: 0, fontWeight: 600 }}>
                  Member of Parliament: <strong style={{ color: '#0F172A', fontWeight: 700 }}>{summary.mp_name}</strong>
                </p>
              </div>
              <div style={{ backgroundColor: '#EFF6FF', border: '1px solid #BFDBFE', padding: '10px 18px', borderRadius: '10px', textAlign: 'center' }}>
                <div style={{ fontSize: '11px', color: '#1E40AF', fontWeight: 700 }}>AI Surveillance Grid</div>
                <div style={{ fontSize: '13.5px', fontWeight: 800, color: '#1D4ED8' }}>Operational (3/3 Models)</div>
              </div>
            </div>

            {/* ================= TAB 1: OVERVIEW DASHBOARD ================= */}
            {activeTab === 'overview' && (
              <>
                {/* 5 KPI Cards Row */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '16px' }}>
                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', padding: '18px', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', display: 'flex', alignItems: 'center', gap: '14px', height: '112px', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <div style={{ width: '46px', height: '46px', backgroundColor: '#EFF6FF', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#2563EB', flexShrink: 0 }}><IndianRupee size={24}/></div>
                    <div>
                      <div style={{ fontSize: '13px', color: '#64748B', fontWeight: 500, marginBottom: '3px' }}>Total MPLADS Funds</div>
                      <div style={{ fontSize: '24px', fontWeight: 700, color: '#0F172A', letterSpacing: '-0.3px' }}>₹ {summary.total_funds} Cr</div>
                      <div style={{ fontSize: '12px', color: '#16A34A', marginTop: '2px', fontWeight: 500 }}>↑ 6.4% vs previous period</div>
                    </div>
                  </div>

                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', padding: '18px', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', display: 'flex', alignItems: 'center', gap: '14px', height: '112px', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <div style={{ width: '46px', height: '46px', backgroundColor: '#F0FDF4', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#16A34A', flexShrink: 0 }}><FileText size={24}/></div>
                    <div>
                      <div style={{ fontSize: '13px', color: '#64748B', fontWeight: 500, marginBottom: '3px' }}>Total Work Orders</div>
                      <div style={{ fontSize: '24px', fontWeight: 700, color: '#0F172A', letterSpacing: '-0.3px' }}>{summary.total_work_orders}</div>
                      <div style={{ fontSize: '12px', color: '#16A34A', marginTop: '2px', fontWeight: 500 }}>↑ 12 new this quarter</div>
                    </div>
                  </div>

                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', padding: '18px', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', display: 'flex', alignItems: 'center', gap: '14px', height: '112px', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <div style={{ width: '46px', height: '46px', backgroundColor: '#F5F3FF', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#8B5CF6', flexShrink: 0 }}><CheckCircle2 size={24}/></div>
                    <div>
                      <div style={{ fontSize: '13px', color: '#64748B', fontWeight: 500, marginBottom: '3px' }}>Completed Projects</div>
                      <div style={{ fontSize: '24px', fontWeight: 700, color: '#0F172A', letterSpacing: '-0.3px' }}>{summary.completed_projects} <span style={{fontSize:'12px', color:'#64748B', fontWeight: 400}}>({summary.completed_pct}%)</span></div>
                      <div style={{ fontSize: '12px', color: '#16A34A', marginTop: '2px', fontWeight: 500 }}>↑ 8% vs previous period</div>
                    </div>
                  </div>

                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', padding: '18px', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', display: 'flex', alignItems: 'center', gap: '14px', height: '112px', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <div style={{ width: '46px', height: '46px', backgroundColor: '#FFF7ED', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#F97316', flexShrink: 0 }}><Activity size={24}/></div>
                    <div>
                      <div style={{ fontSize: '13px', color: '#64748B', fontWeight: 500, marginBottom: '3px' }}>Fund Utilization</div>
                      <div style={{ fontSize: '24px', fontWeight: 700, color: '#0F172A', letterSpacing: '-0.3px' }}>{calculatedUtilization}%</div>
                      <div style={{ fontSize: '12px', color: '#16A34A', marginTop: '2px', fontWeight: 500 }}>↑ 5.3% vs previous period</div>
                    </div>
                  </div>

                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', padding: '18px', borderRadius: '14px', border: '1px solid #FECACA', display: 'flex', alignItems: 'center', gap: '14px', height: '112px', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <div style={{ width: '46px', height: '46px', backgroundColor: '#FEF2F2', borderRadius: '50%', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#EF4444', flexShrink: 0 }}><AlertTriangle size={24}/></div>
                    <div>
                      <div style={{ fontSize: '13px', color: '#64748B', fontWeight: 500, marginBottom: '3px' }}>High Risk Flags</div>
                      <div style={{ fontSize: '24px', fontWeight: 700, color: '#EF4444', letterSpacing: '-0.3px' }}>{summary.high_risk_orders}</div>
                      <div style={{ fontSize: '12px', color: '#EF4444', marginTop: '2px', fontWeight: 500 }}>Requires Attention</div>
                    </div>
                  </div>
                </div>

                {/* 3 Charts Row */}
                <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.2fr 1fr', gap: '20px' }}>
                  
                  {/* Chart 1 */}
                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', padding: '20px', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                      <h3 style={{ fontSize: '17px', fontWeight: 600, margin: 0, color: '#0F172A' }}>Fund Utilization Trend</h3>
                      <div style={{ display: 'flex', gap: '12px', fontSize: '13px', fontWeight: 500 }}>
                        <span style={{ color: '#2563EB' }}>● Sanctioned</span> <span style={{ color: '#16A34A' }}>● Expenditure</span>
                      </div>
                    </div>
                    <div style={{ height: '240px', width: '100%' }}>
                      <ResponsiveContainer>
                        <LineChart data={trend_data} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                          <XAxis dataKey="year" tick={{fontSize: 12, fill: '#64748b'}} axisLine={{stroke: '#E2E8F0'}} tickLine={false} />
                          <YAxis tick={{fontSize: 12, fill: '#64748b'}} tickFormatter={(val) => `${val} Cr`} axisLine={false} tickLine={false} />
                          <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)', fontSize: '13px' }} />
                          <Line type="monotone" dataKey="sanctioned" stroke="#2563EB" strokeWidth={2.5} dot={{r: 3}} />
                          <Line type="monotone" dataKey="expenditure" stroke="#16A34A" strokeWidth={2.5} dot={{r: 3}} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Chart 2 */}
                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', padding: '20px', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <h3 style={{ fontSize: '17px', fontWeight: 600, margin: '0 0 14px 0', color: '#0F172A' }}>Risk Distribution (AI Analysis)</h3>
                    <div style={{ display: 'flex', alignItems: 'center', height: '240px' }}>
                      <div style={{ width: '160px', height: '160px', position: 'relative' }}>
                        <ResponsiveContainer>
                          <PieChart>
                            <Pie data={risk_distribution} innerRadius={55} outerRadius={78} dataKey="value" stroke="none" paddingAngle={2}>
                              {risk_distribution.map((entry, index) => <Cell key={`cell-${index}`} fill={entry.fill} />)}
                            </Pie>
                          </PieChart>
                        </ResponsiveContainer>
                        <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                          <div style={{ fontSize: '26px', fontWeight: 700, color: '#0F172A' }}>{summary.total_work_orders}</div>
                          <div style={{ fontSize: '11px', color: '#64748B', textAlign: 'center', fontWeight: 400 }}>Total Work Orders</div>
                        </div>
                      </div>
                      <div style={{ flex: 1, paddingLeft: '14px', display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '13px' }}>
                        {risk_distribution.map(r => (
                          <div key={r.name} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <span style={{ color: '#475569', fontWeight: 500 }}><span style={{ color: r.fill, fontSize: '13px', marginRight: '6px' }}>●</span>{r.name}</span>
                            <span style={{ fontWeight: 600, color: '#0F172A' }}>{Math.round((r.value/summary.total_work_orders)*100)}%</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Chart 3 */}
                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', padding: '20px', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                      <h3 style={{ fontSize: '17px', fontWeight: 600, margin: 0, color: '#0F172A' }}>Risk Trend</h3>
                      <span style={{ fontSize: '12px', color: '#0F172A', border: '1px solid #CBD5E1', padding: '4px 10px', borderRadius: '6px', fontWeight: 500, backgroundColor: '#F8FAFC' }}>This Year ▾</span>
                    </div>
                    <div style={{ height: '240px', width: '100%' }}>
                      <ResponsiveContainer>
                        <LineChart data={risk_trend} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                          <XAxis dataKey="month" tick={{fontSize: 12, fill: '#64748b'}} axisLine={false} tickLine={false} />
                          <YAxis tick={{fontSize: 12, fill: '#64748b'}} tickFormatter={(val) => `${val}%`} axisLine={false} tickLine={false} />
                          <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)', fontSize: '13px' }} />
                          <Line type="monotone" dataKey="high" stroke="#EF4444" strokeWidth={2} dot={{r: 3}} />
                          <Line type="monotone" dataKey="medium" stroke="#F97316" strokeWidth={2} dot={{r: 3}} />
                          <Line type="monotone" dataKey="low" stroke="#10B981" strokeWidth={2} dot={{r: 3}} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                </div>

                {/* Table & Overall Risk Component */}
                <div style={{ display: 'grid', gridTemplateColumns: '2.5fr 1fr', gap: '20px' }}>
                  
                  {/* Enhanced Work Order Table */}
                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', overflow: 'hidden', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <div style={{ padding: '18px 24px', borderBottom: '1px solid #E2E8F0', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <h3 style={{ fontSize: '17px', fontWeight: 600, margin: 0, color: '#0F172A' }}>Top Flagged Work Orders</h3>
                      <button onClick={() => setActiveTab('workorders')} style={{ backgroundColor: '#2563EB', color: '#FFF', border: 'none', padding: '8px 16px', borderRadius: '6px', fontSize: '12px', fontWeight: 600, cursor: 'pointer' }}>View All Work Orders →</button>
                    </div>
                    
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'center', fontSize: '13px' }}>
                        <thead style={{ color: '#0F172A', fontWeight: 600 }}>
                          <tr style={{ backgroundColor: 'rgba(248, 250, 252, 0.8)' }}>
                            <th style={{ padding: '14px 12px', borderBottom: '1px solid #E2E8F0', fontSize: '12px' }}>ID</th>
                            <th style={{ padding: '14px 12px', borderBottom: '1px solid #E2E8F0', textAlign: 'left', fontSize: '12px' }}>Scheme Description</th>
                            <th style={{ padding: '14px 12px', borderBottom: '1px solid #E2E8F0', fontSize: '12px' }}>Rec. Date</th>
                            <th style={{ padding: '14px 12px', borderBottom: '1px solid #E2E8F0', fontSize: '12px' }}>Amount</th>
                            <th style={{ padding: '14px 12px', borderBottom: '1px solid #E2E8F0', fontSize: '12px' }}>Progress</th>
                            <th style={{ padding: '14px 12px', borderBottom: '1px solid #E2E8F0', fontSize: '12px' }}>AI Risk Score</th>
                            <th style={{ padding: '14px 12px', borderBottom: '1px solid #E2E8F0', fontSize: '12px' }}>Status</th>
                            <th style={{ padding: '14px 12px', borderBottom: '1px solid #E2E8F0', fontSize: '12px' }}>Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {projects.map((proj, idx) => {
                            const sub = getSubScores(proj);
                            return (
                            <tr key={idx} style={{ borderBottom: '1px solid #F1F5F9', height: '64px', transition: 'background 0.15s ease' }} onMouseOver={e => e.currentTarget.style.backgroundColor='rgba(248, 250, 252, 0.9)'} onMouseOut={e => e.currentTarget.style.backgroundColor='transparent'}>
                              <td style={{ padding: '12px', fontWeight: 600, color: '#0F172A', fontSize: '12px' }}>{proj.work_id}</td>
                              <td style={{ padding: '12px', textAlign: 'left', fontWeight: 500, color: '#334155', fontSize: '13px' }}>
                                {proj.description}
                                <div style={{ fontSize: '11px', color: '#64748B', fontWeight: 400, marginTop: '2px', display: 'flex', alignItems: 'center', gap: '4px' }}><MapPin size={10}/> {proj.location || "Block Area"}</div>
                              </td>
                              <td style={{ padding: '12px', color: '#475569', fontSize: '12px', fontWeight: 400 }}>{sub.recDate}</td>
                              <td style={{ padding: '12px', fontWeight: 600, color: '#0F172A', fontSize: '13px' }}>₹ {proj.amount}</td>
                              
                              <td style={{ padding: '12px', width: '110px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '3px', fontWeight: 500, color: '#334155' }}><span>{sub.progress}%</span></div>
                                <div style={{ width: '100%', height: '6px', backgroundColor: '#E2E8F0', borderRadius: '4px' }}>
                                  <div style={{ width: `${sub.progress}%`, height: '100%', backgroundColor: sub.progress === 100 ? '#16A34A' : '#2563EB', borderRadius: '4px' }}></div>
                                </div>
                              </td>

                              <td style={{ padding: '12px' }}>
                                <div style={{ backgroundColor: proj.risk === 'High' ? '#FEF2F2' : proj.risk === 'Medium' ? '#FFF7ED' : '#F0FDF4', color: proj.risk === 'High' ? '#EF4444' : proj.risk === 'Medium' ? '#EA580C' : '#16A34A', padding: '5px 10px', borderRadius: '6px', display: 'inline-flex', flexDirection: 'column', alignItems: 'center', border: `1px solid ${proj.risk === 'High' ? '#FECACA' : '#FED7AA'}` }}>
                                  <div style={{ fontWeight: 600, fontSize: '13px' }}>{proj.overall}</div>
                                  <div style={{ fontSize: '11px', fontWeight: 500 }}>{proj.risk}</div>
                                </div>
                              </td>
                              <td style={{ padding: '12px' }}>
                                <span style={{ fontSize: '11px', fontWeight: 500, color: proj.status === 'Completed' ? '#16A34A' : '#D97706', backgroundColor: proj.status === 'Completed' ? '#DCFCE7' : '#FEF3C7', padding: '5px 10px', borderRadius: '4px' }}>{proj.status}</span>
                              </td>
                              <td style={{ padding: '12px' }}>
                                <button 
                                  onClick={() => setSelectedProjectDetails(proj)}
                                  style={{ border: '1px solid #CBD5E1', backgroundColor: '#FFFFFF', color: '#2563EB', padding: '6px 12px', borderRadius: '6px', fontSize: '11px', cursor: 'pointer', fontWeight: 500, boxShadow: '0 1px 2px rgba(0,0,0,0.02)' }}>
                                  View Details
                                </button>
                              </td>
                            </tr>
                          )})}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Overall Risk Gauge */}
                  <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', borderRadius: '14px', border: '1px solid rgba(226, 232, 240, 0.8)', padding: '22px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                    <h3 style={{ fontSize: '17px', fontWeight: 600, margin: '0 0 16px 0', color: '#0F172A', alignSelf: 'flex-start' }}>Overall Risk Overview</h3>
                    
                    <div style={{ position: 'relative', width: '210px', height: '115px', overflow: 'hidden' }}>
                      <svg viewBox="0 0 200 120" width="100%" height="100%">
                        <defs>
                          <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#10B981" />
                            <stop offset="50%" stopColor="#FACC15" />
                            <stop offset="100%" stopColor="#EF4444" />
                          </linearGradient>
                        </defs>
                        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#E2E8F0" strokeWidth="16" strokeLinecap="round" />
                        <path 
                          d="M 20 100 A 80 80 0 0 1 180 100" 
                          fill="none" 
                          stroke="url(#gaugeGradient)" 
                          strokeWidth="16" 
                          strokeLinecap="round"
                          strokeDasharray={circumference}
                          strokeDashoffset={strokeOffset}
                          style={{ transition: 'stroke-dashoffset 1s ease-in-out' }}
                        />
                      </svg>
                      <div style={{ position: 'absolute', bottom: '0px', left: '0', width: '100%', textAlign: 'center' }}>
                        <div style={{ fontSize: '36px', fontWeight: 700, color: gaugeColor, lineHeight: '1', letterSpacing: '-0.5px' }}>{avgRisk}</div>
                        <div style={{ fontSize: '13px', fontWeight: 600, color: gaugeColor, marginTop: '6px' }}>{riskLabel}</div>
                      </div>
                    </div>
                    <div style={{ fontSize: '12px', color: '#64748B', textAlign: 'center', marginTop: '16px', lineHeight: '1.4', fontWeight: 400 }}>
                      Constituency risk is AI-calculated based on MPLADS work order analysis.
                    </div>
                  </div>
                </div>
              </>
            )}

            {/* ================= TAB 2: WORK ORDERS ================= */}
            {activeTab === 'workorders' && (
              <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', borderRadius: '14px', padding: '28px', border: '1px solid rgba(226, 232, 240, 0.8)', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                  <div>
                    <h2 style={{ fontSize: '18px', fontWeight: 600, margin: '0 0 4px 0', color: '#0F172A' }}>Work Orders Ledger</h2>
                    <p style={{ fontSize: '13px', color: '#64748b', margin: 0, fontWeight: 400 }}>Showing comprehensive list of sanctioned projects for {selectedConstituency}.</p>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', border: '1px solid #CBD5E1', padding: '0 12px', height: '40px', borderRadius: '8px', width: '300px', backgroundColor: '#F8FAFC' }}>
                    <Search size={16} color="#64748b" />
                    <input type="text" placeholder="Search ID or description..." value={workOrderSearch} onChange={(e) => setWorkOrderSearch(e.target.value)} style={{ border: 'none', outline: 'none', width: '100%', background: 'transparent', fontSize: '13px' }} />
                  </div>
                </div>
                <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '13px' }}>
                  <thead>
                    <tr style={{ backgroundColor: 'rgba(248, 250, 252, 0.8)', borderBottom: '2px solid #E2E8F0', color: '#0F172A', height: '44px' }}>
                      <th style={{ padding: '12px', fontWeight: 600 }}>Work Order ID</th>
                      <th style={{ padding: '12px', fontWeight: 600 }}>Description</th>
                      <th style={{ padding: '12px', fontWeight: 600 }}>Amount</th>
                      <th style={{ padding: '12px', fontWeight: 600 }}>Risk Status</th>
                      <th style={{ padding: '12px', fontWeight: 600 }}>Status</th>
                      <th style={{ padding: '12px', fontWeight: 600, textAlign: 'center' }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredProjects.map((p, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #F1F5F9', height: '60px' }}>
                        <td style={{ padding: '12px', fontWeight: 600, color: '#0F172A' }}>{p.work_id}</td>
                        <td style={{ padding: '12px', color: '#334155', fontWeight: 400 }}>{p.description}</td>
                        <td style={{ padding: '12px', fontWeight: 600, color: '#0F172A' }}>₹ {p.amount}</td>
                        <td style={{ padding: '12px', color: p.risk === 'High' ? '#EF4444' : p.risk === 'Medium' ? '#F97316' : '#10B981', fontWeight: 600 }}>{p.risk} ({p.overall})</td>
                        <td style={{ padding: '12px' }}>
                          <span style={{ fontSize: '12px', fontWeight: 500, color: p.status === 'Completed' ? '#16A34A' : '#D97706', backgroundColor: p.status === 'Completed' ? '#DCFCE7' : '#FEF3C7', padding: '4px 8px', borderRadius: '4px' }}>{p.status}</span>
                        </td>
                        <td style={{ padding: '12px', textAlign: 'center' }}>
                          <button onClick={() => setSelectedProjectDetails(p)} style={{ padding: '6px 12px', backgroundColor: '#2563EB', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 500, fontSize: '12px' }}>Inspect Audit</button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* ================= TAB 3: RISK ALERTS ================= */}
            {activeTab === 'riskalerts' && (
              <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', borderRadius: '14px', padding: '28px', border: '1px solid rgba(226, 232, 240, 0.8)', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 600, marginBottom: '6px', color: '#EF4444', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <AlertTriangle size={22} /> Active Risk Alerts ({summary.high_risk_orders} Critical)
                </h2>
                <p style={{ fontSize: '13px', color: '#64748B', marginBottom: '24px', fontWeight: 400 }}>Projects flagged by Machine Learning isolation models requiring immediate attention.</p>
                
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {projects.filter(p => p.risk === 'High' || p.risk === 'Medium').map((proj, idx) => (
                    <div key={idx} style={{ padding: '18px', border: `1px solid ${proj.risk === 'High' ? '#FECACA' : '#FED7AA'}`, backgroundColor: proj.risk === 'High' ? '#FEF2F2' : '#FFF7ED', borderRadius: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ fontWeight: 600, color: proj.risk === 'High' ? '#991B1B' : '#9A3412', fontSize: '14px' }}>{proj.work_id} — {proj.description}</div>
                        <div style={{ fontSize: '13px', color: proj.risk === 'High' ? '#7F1D1D' : '#7C2D12', marginTop: '4px', fontWeight: 500 }}>AI Risk Score: <strong style={{fontWeight: 600}}>{proj.overall} ({proj.risk})</strong></div>
                      </div>
                      <button onClick={() => setSelectedProjectDetails(proj)} style={{ padding: '8px 16px', backgroundColor: proj.risk === 'High' ? '#EF4444' : '#EA580C', color: '#fff', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 600, fontSize: '12px' }}>View Details</button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ================= TAB 4: SCHEME ANALYTICS ================= */}
            {activeTab === 'scheme' && (
              <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', borderRadius: '14px', padding: '28px', border: '1px solid rgba(226, 232, 240, 0.8)', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 600, margin: '0 0 4px 0', color: '#0F172A' }}>Scheme Analytics Module</h2>
                <p style={{ fontSize: '13px', color: '#64748b', margin: '0 0 24px 0', fontWeight: 400 }}>Sector-wise fund allocation vs utilization for {selectedConstituency}.</p>
                <div style={{ height: '320px', width: '100%' }}>
                  <ResponsiveContainer>
                    <BarChart data={schemeData} margin={{ top: 10, right: 20, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0"/>
                      <XAxis dataKey="category" tick={{fontSize: 13, fill: '#475569'}} axisLine={false} tickLine={false} />
                      <YAxis tick={{fontSize: 13, fill: '#475569'}} tickFormatter={(v)=>`₹${v}Cr`} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)', fontSize: '13px' }} />
                      <Legend wrapperStyle={{paddingTop: '10px', fontSize: '13px'}} />
                      <Bar dataKey="allocated" name="Allocated Funds (Cr)" fill="#2563EB" radius={[4, 4, 0, 0]} barSize={38} />
                      <Bar dataKey="utilized" name="Utilized Funds (Cr)" fill="#16A34A" radius={[4, 4, 0, 0]} barSize={38} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* ================= TAB 5: REPORTS ================= */}
            {activeTab === 'reports' && (
              <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', borderRadius: '14px', padding: '28px', border: '1px solid rgba(226, 232, 240, 0.8)', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                <h2 style={{ fontSize: '18px', fontWeight: 600, margin: '0 0 4px 0', color: '#0F172A' }}>Automated Audit Reports</h2>
                <p style={{ fontSize: '13px', color: '#64748b', margin: '0 0 24px 0', fontWeight: 400 }}>Download system-generated PDF reports for compliance and auditing.</p>
                
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '18px' }}>
                  <div style={{ border: '1px solid #E2E8F0', padding: '18px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', backgroundColor: 'rgba(255,255,255,0.95)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                      <div style={{ backgroundColor: '#FEF2F2', padding: '12px', borderRadius: '8px', color: '#EF4444' }}><FileCheck size={24}/></div>
                      <div>
                        <div style={{ fontSize: '14px', fontWeight: 600, color: '#0F172A' }}>Monthly Anomaly Report</div>
                        <div style={{ fontSize: '12px', color: '#64748B', marginTop: '2px' }}>PDF • Generated: Today, 09:00 AM</div>
                      </div>
                    </div>
                    <button onClick={handleDownloadPDF} style={{ padding: '8px 16px', backgroundColor: '#2563EB', color: '#FFF', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '13px' }}>Download</button>
                  </div>
                  <div style={{ border: '1px solid #E2E8F0', padding: '18px', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', backgroundColor: 'rgba(255,255,255,0.95)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                      <div style={{ backgroundColor: '#EFF6FF', padding: '12px', borderRadius: '8px', color: '#2563EB' }}><FileCheck size={24}/></div>
                      <div>
                        <div style={{ fontSize: '14px', fontWeight: 600, color: '#0F172A' }}>Full Constituency Summary</div>
                        <div style={{ fontSize: '12px', color: '#64748B', marginTop: '2px' }}>PDF • Generated: 1 Week Ago</div>
                      </div>
                    </div>
                    <button onClick={handleDownloadPDF} style={{ padding: '8px 16px', backgroundColor: '#2563EB', color: '#FFF', borderRadius: '6px', border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '13px' }}>Download</button>
                  </div>
                </div>
              </div>
            )}

            {/* ================= TAB 6: AUDIT TRAIL ================= */}
            {activeTab === 'audittrail' && (
              <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', borderRadius: '14px', padding: '28px', border: '1px solid rgba(226, 232, 240, 0.8)', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
                  <div>
                    <h2 style={{ fontSize: '18px', fontWeight: 600, margin: '0 0 4px 0', color: '#0F172A' }}>System Audit Trail</h2>
                    <p style={{ fontSize: '13px', color: '#64748b', margin: 0, fontWeight: 400 }}>Immutable logs of AI flags and official interventions.</p>
                  </div>
                  <button style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '8px 14px', backgroundColor: '#FFFFFF', border: '1px solid #CBD5E1', borderRadius: '6px', fontWeight: 500, fontSize: '13px', cursor: 'pointer' }}><RefreshCw size={15}/> Sync Logs</button>
                </div>

                <div style={{ paddingLeft: '18px', borderLeft: '2px solid #E2E8F0', display: 'flex', flexDirection: 'column', gap: '20px', fontSize: '13px' }}>
                  <div style={{ position: 'relative' }}>
                    <div style={{ position: 'absolute', left: '-23px', top: '2px', width: '9px', height: '9px', backgroundColor: '#EF4444', borderRadius: '50%' }}></div>
                    <div style={{ fontSize: '12px', color: '#64748B', fontWeight: 600 }}>Today, 11:30 AM</div>
                    <div style={{ fontSize: '14px', color: '#0F172A', fontWeight: 600, marginTop: '2px' }}>AI Model Flagged Work Order</div>
                    <div style={{ fontSize: '13px', color: '#475569', marginTop: '2px' }}>Anomaly detected in semantic project description.</div>
                  </div>
                  <div style={{ position: 'relative' }}>
                    <div style={{ position: 'absolute', left: '-23px', top: '2px', width: '9px', height: '9px', backgroundColor: '#2563EB', borderRadius: '50%' }}></div>
                    <div style={{ fontSize: '12px', color: '#64748B', fontWeight: 600 }}>Yesterday, 04:15 PM</div>
                    <div style={{ fontSize: '14px', color: '#0F172A', fontWeight: 600, marginTop: '2px' }}>District Authority Requested Clarification</div>
                    <div style={{ fontSize: '13px', color: '#475569', marginTop: '2px' }}>Query sent to nodal agency regarding milestone delay.</div>
                  </div>
                </div>
              </div>
            )}

            {/* ================= OTHER TABS ================= */}
            {['constituency', 'users', 'settings'].includes(activeTab) && (
              <div style={{ backgroundColor: 'rgba(255, 255, 255, 0.94)', backdropFilter: 'blur(8px)', borderRadius: '14px', padding: '40px', border: '1px solid rgba(226, 232, 240, 0.8)', textAlign: 'center', boxShadow: '0 2px 6px rgba(0,0,0,0.02)' }}>
                <h2 style={{ fontSize: '20px', fontWeight: 600, textTransform: 'capitalize', marginBottom: '8px', color: '#0F172A' }}>{activeTab} Module</h2>
                <p style={{ fontSize: '14px', color: '#64748B', marginBottom: '28px', fontWeight: 400 }}>Advanced telemetry and configurations for {selectedConstituency} are linked securely to the backend grid.</p>
                <button onClick={() => alert("Diagnostic sync successful.")} style={{ padding: '10px 20px', backgroundColor: '#2563EB', color: '#fff', border: 'none', borderRadius: '6px', fontWeight: 600, cursor: 'pointer', fontSize: '13px' }}>Run Module Diagnostics</button>
              </div>
            )}

          </div>
        </div>
      </div>

      {/* --- AI ANALYSIS MODAL POPUP --- */}
      {selectedProjectDetails && (
        <div style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, backgroundColor: 'rgba(15, 23, 42, 0.6)', display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 9999, backdropFilter: 'blur(3px)' }}>
          <div style={{ backgroundColor: '#fff', width: '660px', borderRadius: '14px', overflow: 'hidden', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)' }}>
            <div style={{ backgroundColor: '#0A192F', padding: '22px 28px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: '#fff' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <BrainCircuit color="#38BDF8" size={24} />
                <h3 style={{ margin: 0, fontSize: '17px', fontWeight: 600 }}>AI Risk Analysis & SHAP Explainability</h3>
              </div>
              <button onClick={() => setSelectedProjectDetails(null)} style={{ background: 'transparent', border: 'none', color: '#94A3B8', cursor: 'pointer' }}><X size={24} /></button>
            </div>
            
            <div style={{ padding: '28px' }}>
              {(() => {
                const sub = getSubScores(selectedProjectDetails);
                return (
                  <>
                    <div style={{ marginBottom: '28px', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                      <div>
                        <div style={{ fontSize: '12px', color: '#64748B', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Work Order ID: {selectedProjectDetails.work_id}</div>
                        <div style={{ fontSize: '17px', fontWeight: 600, color: '#0F172A', marginTop: '4px' }}>{selectedProjectDetails.description}</div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '12px', color: '#64748B', fontWeight: 600 }}>Rec. Date: {sub.recDate}</div>
                        <div style={{ fontSize: '14px', fontWeight: 600, color: '#0F172A', marginTop: '2px' }}>₹ {selectedProjectDetails.amount}</div>
                      </div>
                    </div>
                    
                    <div style={{ display: 'flex', gap: '16px', marginBottom: '28px' }}>
                      <div style={{ flex: 1, backgroundColor: '#F8FAFC', padding: '18px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                        <div style={{ fontSize: '13px', color: '#64748B', marginBottom: '4px', fontWeight: 500 }}>Overall Risk Score</div>
                        <div style={{ fontSize: '28px', fontWeight: 700, color: selectedProjectDetails.risk === 'High' ? '#EF4444' : selectedProjectDetails.risk === 'Medium' ? '#F97316' : '#10B981' }}>
                          {(selectedProjectDetails.overall * 100).toFixed(0)}%
                        </div>
                      </div>
                      
                      <div style={{ flex: 1, backgroundColor: '#F8FAFC', padding: '18px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                        <div style={{ fontSize: '13px', color: '#64748B', marginBottom: '4px', fontWeight: 500 }}>Physical Progress</div>
                        <div style={{ fontSize: '28px', fontWeight: 700, color: '#2563EB' }}>{sub.progress}%</div>
                      </div>

                      <div style={{ flex: 1.5, backgroundColor: '#F8FAFC', padding: '18px', borderRadius: '8px', border: '1px solid #E2E8F0' }}>
                        <div style={{ fontSize: '13px', color: '#64748B', marginBottom: '4px', fontWeight: 500 }}>AI Flag Status</div>
                        <div style={{ fontSize: '16px', fontWeight: 600, color: selectedProjectDetails.risk === 'High' ? '#EF4444' : selectedProjectDetails.risk === 'Medium' ? '#F97316' : '#10B981', marginTop: '6px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <ShieldAlert size={20} /> {selectedProjectDetails.risk} Risk Detected
                        </div>
                      </div>
                    </div>

                    <h4 style={{ fontSize: '16px', fontWeight: 600, borderBottom: '1px solid #E2E8F0', paddingBottom: '12px', marginBottom: '18px', color: '#0F172A' }}>Risk Driver Breakdown (SHAP Values)</h4>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '18px', fontSize: '13px' }}>
                      <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 500, marginBottom: '6px', color: '#334155' }}>
                            <span>Semantic Duplication Probability</span>
                            <span style={{fontWeight: 600}}>{(sub.dup * 100).toFixed(0)}%</span>
                          </div>
                          <div style={{ width: '100%', height: '8px', backgroundColor: '#E2E8F0', borderRadius: '4px' }}>
                            <div style={{ width: `${sub.dup * 100}%`, height: '100%', backgroundColor: '#EF4444', borderRadius: '4px' }}></div>
                          </div>
                      </div>
                      
                      <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 500, marginBottom: '6px', color: '#334155' }}>
                            <span>Cost Overrun / Delay Risk</span>
                            <span style={{fontWeight: 600}}>{(sub.del * 100).toFixed(0)}%</span>
                          </div>
                          <div style={{ width: '100%', height: '8px', backgroundColor: '#E2E8F0', borderRadius: '4px' }}>
                            <div style={{ width: `${sub.del * 100}%`, height: '100%', backgroundColor: '#F97316', borderRadius: '4px' }}></div>
                          </div>
                      </div>
                      
                      <div>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontWeight: 500, marginBottom: '6px', color: '#334155' }}>
                            <span>Compliance Irregularity</span>
                            <span style={{fontWeight: 600}}>{(sub.comp * 100).toFixed(0)}%</span>
                          </div>
                          <div style={{ width: '100%', height: '8px', backgroundColor: '#E2E8F0', borderRadius: '4px' }}>
                            <div style={{ width: `${sub.comp * 100}%`, height: '100%', backgroundColor: '#2563EB', borderRadius: '4px' }}></div>
                          </div>
                      </div>
                    </div>
                  </>
                );
              })()}
            </div>
          </div>
        </div>
      )}

    </div>
  
  );
}