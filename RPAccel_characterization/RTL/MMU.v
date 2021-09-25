`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Harvard University
// Jeff (Jun) Zhang
// jeffzhang@g.harvard.edu 

// Create Date:    18:34:05 04/22/2021 
// Design Name: 
// Module Name:    MMU 
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
 

module MMU #(parameter depth=16, bit_width=16, acc_width=64, size=16)
(
	clk,
	control,
	data_arr, //mux
	wt_arr,
	acc_in, 
	data_arr_out,
	wt_arr_out,
	acc_out  //mux
    );


input clk;
input control;
input  [(bit_width*depth)-1:0] data_arr;
input  [(bit_width*depth)-1:0] wt_arr;
input [acc_width*size-1 : 0] acc_in;  
output wire [(bit_width*depth)-1:0] data_arr_out;  //double check
output wire [(bit_width*depth)-1:0] wt_arr_out;
output reg [acc_width*size-1 : 0] acc_out;
	


	generate 
	genvar i;
	
	for(i=0;i<depth+1;i=i+1)
	begin:temp_acc
	
	wire [acc_width*size-1:0] test;
	end
	
	endgenerate
	
	
	//assign temp_acc[0].test=0;
	assign temp_acc[0].test=acc_in;

	always@(*)begin
	//if(~control)
	acc_out=temp_acc[depth].test;
	//else
	//acc_out=0;
	end


	
	generate 
	//genvar i;

	for(i=0;i<depth;i=i+1)
		begin:chain	
	//	temp_acc[]
		MAC_chain #(.size(size),.bit_width(bit_width),.acc_width_curr(acc_width),.acc_width_next(acc_width)) chain(
			.clk(clk),
			.control(control),
			.data(data_arr[(depth-1-i)*bit_width+bit_width-1 : (depth-1-i)*bit_width]),
			.weight(wt_arr[(depth-1-i)*bit_width+bit_width-1 : (depth-1-i)*bit_width]),
			.acc_in(temp_acc[i].test),
			.data_out(data_arr_out[(depth-1-i)*bit_width+bit_width-1 : (depth-1-i)*bit_width]),
			.weight_out(wt_arr_out[(depth-1-i)*bit_width+bit_width-1 : (depth-1-i)*bit_width]),
			.acc_out(temp_acc[i+1].test));

		end
		
	endgenerate




endmodule
