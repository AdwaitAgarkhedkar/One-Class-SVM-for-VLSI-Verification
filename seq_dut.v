// Overlapping sequence detector for the pattern 110101 (Moore ).
module seq_dut (
    input  clk,
    input  rst,
    input  din,
    output detected
);

    parameter S0 = 3'd0;   
    parameter S1 = 3'd1;   
    parameter S2 = 3'd2;  
    parameter S3 = 3'd3;   
    parameter S4 = 3'd4;   
    parameter S5 = 3'd5;   
    parameter S6 = 3'd6;   
    reg [2:0] state, next_state;

    always @(posedge clk) begin
        if (rst) state <= S0;
        else     state <= next_state;
    end

    always @(*) begin
        case (state)
            S0:      next_state = din ? S1 : S0;
            S1:      next_state = din ? S2 : S0;
            S2:      next_state = din ? S2 : S3;
            S3:      next_state = din ? S4 : S0;
            S4:      next_state = din ? S2 : S5;
            S5:      next_state = din ? S6 : S0;
            S6:      next_state = din ? S2 : S0;   // overlap
            default: next_state = S0;
        endcase
    end

    assign detected = (state == S6);

endmodule
