window.echarts = echarts;

// 检查echarts和mermaid是否正确加载，如果没有加载，则从本地加载
(function() {
  // 等待DOM加载完成
  document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
      // 检查echarts是否已加载
      if (typeof echarts === 'undefined' || typeof echarts.init !== 'function') {
        console.log('ECharts not loaded from CDN, loading local copy...');
        loadScript('/maxblog/lib/echarts/echarts.min.js', function() {
          console.log('Local ECharts loaded');
          window.echarts = echarts;
          initializeECharts();
        });
      } else {
        console.log('ECharts loaded from CDN');
        window.echarts = echarts;
      }
      
      // 检查mermaid是否已加载
      if (typeof mermaid === 'undefined') {
        console.log('Mermaid not loaded from CDN, loading local copy...');
        loadScript('/maxblog/lib/mermaid/mermaid.min.js', function() {
          console.log('Local Mermaid loaded');
          if (typeof mermaid !== 'undefined') {
            initializeMermaid();
          }
        });
      } else {
        console.log('Mermaid loaded from CDN');
      }
    }, 1000); // 延迟1秒，确保其他脚本已加载
  });
  
  // 加载脚本辅助函数
  function loadScript(url, callback) {
    var script = document.createElement('script');
    script.src = url;
    script.onload = callback;
    script.onerror = function() {
      console.error('Failed to load script:', url);
    };
    document.head.appendChild(script);
  }
  
  // 初始化mermaid
  function initializeMermaid() {
    if (typeof mermaid === 'undefined') {
      console.error('Failed to load Mermaid');
      return;
    }
    
    try {
      // 配置mermaid
      mermaid.initialize({
        startOnLoad: true,
        theme: 'default'
      });
      
      // 手动渲染所有mermaid图表
      mermaid.init(undefined, document.querySelectorAll('.mermaid'));
      console.log('Mermaid initialized');
    } catch (e) {
      console.error('Error initializing Mermaid:', e);
    }
  }
  
  function initializeECharts() {
    if (typeof echarts === 'undefined') {
      console.error('Failed to load ECharts');
      return;
    }
    
    // 获取所有echarts容器
    var charts = document.getElementsByClassName('echarts');
    
    // 遍历容器并初始化图表
    for (var i = 0; i < charts.length; i++) {
      var chartContainer = charts[i];
      var chartId = chartContainer.id;
      
      if (!chartId || !window.config || !window.config.data || !window.config.data[chartId]) {
        console.error('Missing chart data for', chartId);
        continue;
      }
      
      try {
        var chart = echarts.init(chartContainer);
        var options = JSON.parse(window.config.data[chartId]);
        chart.setOption(options);
        console.log('Chart initialized:', chartId);
      } catch (e) {
        console.error('Error initializing chart', chartId, e);
      }
    }
  }
})();
