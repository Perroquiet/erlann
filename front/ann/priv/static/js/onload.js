	function openDialogUpload(){
		$('#basic-modal-content').modal();
	}

var uploadHandler = {


}
	
var csvParser = {

	data: new Array(),

	upload_file: function(file){
	
	
	}

}

var chartHandler = {

	chartObj: "",
	clearChart: function(){
		var chart = this.chartObj;
		chart.hide('fade', 500).html("");
	},
	drawChart: function(cat, data){

    	var options = {
            chart: { renderTo: 'graphic_container', width: 900, height: 500, type: 'line' },
            title: { text: 'Forecast Graph' },
            subtitle: { text: 'Stock Market Forecasting Graph' },
            xAxis: {
				title: { text: 'Days', },
                categories: cat
            },
            yAxis: {
                title: { text: 'Close ' }
            },
			
            tooltip: {
                formatter: function() {
                        return '<b>'+ this.series.name +'</b><br/>'+
                        this.x +': '+ this.y +'';
                }
            }
        };

        //Initializing a new chart with the options above
        chart = new Highcharts.Chart(options);

        //Inserting the data
        chart.addSeries ({data:getDataC1FromCSV(),name:'Hi'},true);
        //chart.addSeries({data:getDataC2FromCSV(),name:'Lo'},true);
    
	}
}

	jQuery(function ($) {
		//Initialize data handlers...
		chartHandler.chartObj = $('#graphic_container');
	
		// Clear graph and Plot graph	
		$('#plotGraph').click(function(){  createMyChart(); });
	    $('#clearGraph').click(function(){ chartHandler.clearChart();  }); //clearChart();
		if(isAPIAvailable()) {
			var finalVector = null;
      		$('#files').bind('change', handleFileSelect);
      		var generalFile = null;
      		
    	}
		
		$('#options-container').hide(0);
		$('#start-date').datepicker({ changeMonth:true, changeYear:true });
		$('#end-date').datepicker({ changeMonth:true, changeYear:true });
		$('#update-options').click(function(){
			
			updateMyChart($('#start-date').val(), $('#end-date').val());
		});
	});	