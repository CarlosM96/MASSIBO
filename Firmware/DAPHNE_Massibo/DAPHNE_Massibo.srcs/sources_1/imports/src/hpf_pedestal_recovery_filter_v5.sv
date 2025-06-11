`timescale 1ns/10ps
//////////////////////////////////////////////////////////////////////////////////
// University: UNIMIB 
// Engineer: Esteban Cristaldo, MSc
//
// Create Date: July 14, 2022, 11:53:42 AM
// Design Name: filtering_and_selftrigger
// Module Name: filtroIIR_integrator_optimized
// Project Name: selftrigger@bicocca
// Target Devices: DAPHNE V1
//
//////////////////////////////////////////////////////////////////////////////////
module hpf_pedestal_recovery_filter_v5(
	input wire clk,
	//input wire fclk
	input wire reset,
    input wire n_1_reset,
	input wire enable,
    input wire [1:0] output_selector,
	//input wire signed [719:0] x,
	input wire signed [143:0] x,
    output wire [7:0] trigger_output,
    //output wire [39:0] trigger_output,
	//output wire signed [719:0] y
	output wire signed [143:0] y
);
	

	wire signed [15:0] lpf_out [4:0][7:0];
	wire signed [15:0] hpf_out [4:0][7:0];
    wire signed [15:0] movmean_out [4:0][7:0];
	wire signed [15:0] x_i [4:0][7:0];
    //wire signed [15:0] w_resta_out [4:0][7:0];
    wire signed [15:0] w_out [4:0][7:0];
	wire signed [15:0] resta_out [4:0][7:0];
	wire signed [15:0] suma_out [4:0][7:0];
    
	generate genvar i,j;
		//for(i=0; i<=4; i=i+1) begin : i_instance
		for(i=0; i<1; i=i+1) begin : i_instance
		    assign y[((i*9 + 8)*16 + 15) : ((i*9 + 8)*16)] = x[((i*9 + 8)*16 + 15) : ((i*9 + 8)*16)]; // (i*9 + j)*16
            for(j=0; j<=7; j=j+1) begin : j_instance
                //if(i == 1 && j == 0) begin // comment to have 40 channels
                filtroIIR_V2 lpf(
                    .clk(clk),
                    .reset(reset),
                    .enable(enable),
                    .x(x_i[i][j]),
                    .y(lpf_out[i][j])
                );

                filtroIIR_integrator_optimized hpf(
                    .clk(clk),
                    .reset(reset),
                    .n_1_reset(n_1_reset),
                    .enable(enable),
                    .x(resta_out[i][j]),
                    //.x(x_i[i][j]),
                    .y(hpf_out[i][j])
                );

                filtroIIR_movmean40_cfd filter_trigger(
                    .clk(clk),
                    .reset(reset),
                    .n_1_reset(n_1_reset),
                    .enable(enable),
                    .x(hpf_out[i][j]),
                    .trigger(trigger_output[i*8 + j]),
                    .y(movmean_out[i][j])
                );

                /*always @(*) begin
                    if(en) begin
                        suma_out[i][j] <= (hpf_out[i][j] + lpf_out[i][j]);
                        resta_out[i][j] <= (x_i[i][j] - lpf_out[i][j]);
                    end else begin
                        suma_out[i][j] <= hpf_out[i][j];
                        resta_out[i][j] <= x_i[i][j];
                    end
                end*/
                assign resta_out[i][j] = (enable==0) ?   x_i[i][j] : 
                                         (enable==1) ?   (x_i[i][j] - lpf_out[i][j]) : 
                                         16'bx;
                
                assign suma_out[i][j] = (enable==0) ?   hpf_out[i][j] : 
                                        (enable==1) ?   (hpf_out[i][j] + lpf_out[i][j]) : 
                                         16'bx;

                assign w_out[i][j] =    (output_selector == 2'b00) ?   suma_out[i][j] : 
                                        (output_selector == 2'b01) ?   movmean_out[i][j] : 
                                        (output_selector == 2'b10) ?   hpf_out[i][j] :
                                        (output_selector == 2'b11) ?   x_i[i][j] :
                                         16'bx;
               
                /*assign suma_out[i][j] = (en==0) ?   hpf_out[i][j] : 
                                        (en==1) ?   (hpf_out[i][j] + 16'd8000) : 
                                         16'bx;*/

                assign x_i[i][j] = x[((i*9 + j)*16 + 15) : ((i*9 + j)*16)];
                assign y[((i*9 + j)*16 + 15) : ((i*9 + j)*16)] = w_out[i][j];
                //assign w_resta_out[i][j] = resta_out[i][j];
                /*end else begin // comment to have 40 channels
                    assign y[((i*9 + j)*16 + 15) : ((i*9 + j)*16)] = x[((i*9 + j)*16 + 15) : ((i*9 + j)*16)];
                end*/
            end
		end
		
	endgenerate
	
endmodule