`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Harvard University
// Jeff (Jun) Zhang
// jeffzhang@g.harvard.edu 

// Create Date:    18:34:05 04/22/2021  
// Design Name: 
// Module Name:    MAC_chain 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module MAC_chain#(parameter size=4, bit_width=8 ,acc_width_curr=32, acc_width_next=32)(
	clk,
	control,
	data,
	weight,
	acc_in,
	data_out,
	weight_out,
    acc_out
	//acc_width_temp
    );
	
	input clk;
	input control;
	input signed [bit_width-1:0] data;
	input signed [bit_width-1:0] weight;
	input signed [acc_width_curr*size-1:0] acc_in;
	//input acc_width_temp;
	//integer acc_width=acc_width;
	output reg signed [bit_width-1:0] data_out;
	output reg signed [bit_width-1:0] weight_out;
	output reg signed [acc_width_next*size-1:0] acc_out;

		//reg temp=0;
	//*reg [acc_width-1:0] acc_in_1[0:size-1]; 
		//reg [31:0] acc_out_1[0:size-1];

	integer m,n;
	
	//always@* begin
	//for(m=0;m<size;m=m+1)begin
	//for(n=0;n<acc_width;n=n+1)begin
	//acc_in_1[m][n]<=acc_in[m*acc_width+n];
	//end
	//end
	//end
	
	wire [acc_width_next-1:0] acc_out_temp[0:size-1];
	
	//another method to avoid "sensitivity list error"
		//generate
		//genvar k;
		//for (k=0; k<size; k=k+1)begin:temp 
		//assign acc_out[32*k+31:32*k] = acc_out_temp[k];
		//end
		//endgenerate
		
	always@* begin
	for(m=0;m<size;m=m+1)begin
	for(n=0;n<acc_width_next;n=n+1)begin
	//acc_out_1[m][n]<=acc_out[m*32+n];
	acc_out[m*acc_width_next+n]=acc_out_temp[m][n];
	end
	end
	end
	
	
	reg [bit_width-1:0] temp_first_wt;
	wire [bit_width-1:0]temp_data[size+1:0];		//to connect MACs in the MAC_chain
	wire [bit_width-1:0]temp_wt[size+1:0];
	assign temp_data[0]=data;
	assign temp_wt[0]=temp_first_wt;
	

    
	
    always@(*)begin
	//if(~control)
	data_out=temp_data[size];
	//else
	//acc_out=0;
	end

    always@(*)begin
	//if(~control)
	weight_out=temp_wt[size];
	//else
	//acc_out=0;
	end
	
	always@(*)begin
	if(control) temp_first_wt=weight; 
	else temp_first_wt= 0;
	end
		
	

	generate 
	genvar i;
	
    for(i=0;i<size;i=i+1)
		begin:mac
		MAC #(.size(size), .bit_width(bit_width),.acc_width_curr(acc_width_curr),.acc_width_next(acc_width_next)) mac(
		.clk(clk),
		.control(control),
		.acc_in(acc_in[i*acc_width_curr+acc_width_curr-1:i*acc_width_curr]),
		.acc_out(acc_out_temp[i]),
		.data_in(temp_data[i]),
		.wt_path_in(temp_wt[i]),
		.data_out(temp_data[i+1]),
		.wt_path_out(temp_wt[i+1]));
		
		end
		
	endgenerate
			
	
	
	
endmodule
