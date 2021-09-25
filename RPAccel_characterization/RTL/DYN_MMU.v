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
 

module DYN_MMU #(parameter depth=32, bit_width=16, acc_width=64, size=32, mmu_block=4)
(
	clk,
	control,
	wt_arr,
	data_arr_alt, //mux
	acc_sle_d,
	data_sle_s,
	wt_arr_out,
	data_arr_out,
	acc_in,
	acc_out  //mux
    );

//localparam mmu_block = 16;
localparam d_len  = depth / mmu_block;
localparam s_len = size / mmu_block;

input clk;
input control;
input  [(bit_width*mmu_block*d_len)-1:0] wt_arr;
input  [bit_width*mmu_block*s_len*d_len-1:0] data_arr_alt;
//input  [d_len-1:0] acc_sle_d;
input  [d_len*s_len-1:0] acc_sle_d;
input [s_len*d_len-1:0] data_sle_s;
input [acc_width*mmu_block*s_len-1:0] acc_in;

output wire [(bit_width*mmu_block*d_len)-1:0] wt_arr_out;
output wire [(bit_width*mmu_block*d_len)-1:0] data_arr_out;
output reg [acc_width*mmu_block*s_len-1 : 0] acc_out;	


	generate 
	genvar i;
	
	for(i=0;i<d_len+1;i=i+1)
	begin:temp_acc
	
	wire [acc_width*mmu_block*s_len-1:0] test;
	wire [acc_width*mmu_block*s_len-1:0] test2;

	end
	
	endgenerate
	
assign temp_acc[0].test=acc_in;
	
always@(*)begin
	//if(~control)
	acc_out=temp_acc[d_len].test;
	//else
	//acc_out=0;
	end


//(acc_width*mmu_block)
	


	generate 
	//genvar i;

	for(i=0;i<d_len;i=i+1)
		begin:mmu_chain	
	//	temp_acc[]

	    //assign temp_acc[i].test2=(acc_sle_d[i])? 0: temp_acc[i].test;

		MMU_chain #(.depth(depth),.bit_width(bit_width),.acc_width(acc_width),.size(size), .mmu_block(mmu_block)) mmu_chain(
			.clk(clk),
			.control(control),

			.data_arr_alt(data_arr_alt[(d_len-1-i)*bit_width*mmu_block*s_len+bit_width*mmu_block*s_len-1 : (d_len-1-i)*bit_width*mmu_block*s_len]),
			.data_arr_sle_s (data_sle_s[(d_len-i)*s_len-1:(d_len-i-1)*s_len]),
			.acc_arr_sle_d (acc_sle_d[(d_len-i)*s_len-1:(d_len-i-1)*s_len]),

			.wt_arr(wt_arr[(d_len-1-i)*bit_width*mmu_block+bit_width*mmu_block-1 : (d_len-1-i)*bit_width*mmu_block]),
			.wt_arr_out(wt_arr_out[(d_len-1-i)*bit_width*mmu_block+bit_width*mmu_block-1 : (d_len-1-i)*bit_width*mmu_block]),
			.data_arr_out(data_arr_out[(d_len-1-i)*bit_width*mmu_block+bit_width*mmu_block-1 : (d_len-1-i)*bit_width*mmu_block]),

			.acc_in(temp_acc[i].test),
			.acc_out(temp_acc[i+1].test));

	end
	
	
  
	  
		
	endgenerate




endmodule
