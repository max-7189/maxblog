window.echarts = echarts;

// 检查echarts是否正确加载，如果没有加载，则从本地加载
(function() {
  // 等待DOM加载完成
  document.addEventListener('DOMContentLoaded', function() {
    setTimeout(function() {
      // 检查echarts是否已加载
      if (typeof echarts === 'undefined' || typeof echarts.init !== 'function') {
        console.log('ECharts not loaded from CDN, loading local copy...');
        
        // 创建script标签
        var script = document.createElement('script');
        script.src = '/maxblog/lib/echarts/echarts.min.js';
        script.onload = function() {
          console.log('Local ECharts loaded, initializing charts...');
          initializeECharts();
        };
        document.head.appendChild(script);
      } else {
        console.log('ECharts loaded from CDN');
      }
    }, 1000); // 延迟1秒，确保其他脚本已加载
  });
  
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
