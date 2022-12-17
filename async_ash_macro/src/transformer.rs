use syn::{Block, Expr, Pat, Stmt};

pub trait CommandsTransformer {
    fn transform_block(&mut self, block: &Block) -> Block {
        Block {
            brace_token: block.brace_token.clone(),
            stmts: block
                .stmts
                .iter()
                .map(move |stmt| self.transform_stmt(stmt))
                .collect(),
        }
    }

    fn transform_pattern(&mut self, pat: &Pat) -> Pat {
        match pat {
            Pat::Box(pat) => Pat::Box(syn::PatBox {
                pat: Box::new(self.transform_pattern(&pat.pat)),
                ..pat.clone()
            }),
            Pat::Ident(ident) => Pat::Ident(syn::PatIdent {
                subpat: ident
                    .subpat
                    .as_ref()
                    .map(|(at, subpat)| (at.clone(), Box::new(self.transform_pattern(subpat)))),
                ..ident.clone()
            }),
            Pat::Lit(lit) => Pat::Lit(syn::PatLit {
                expr: Box::new(self.transform_expr(&lit.expr)),
                ..lit.clone()
            }),
            Pat::Or(clause) => Pat::Or(syn::PatOr {
                cases: clause
                    .cases
                    .iter()
                    .map(|pat| self.transform_pattern(pat))
                    .collect(),
                ..clause.clone()
            }),
            Pat::Range(range) => Pat::Range(syn::PatRange {
                lo: Box::new(self.transform_expr(&range.lo)),
                hi: Box::new(self.transform_expr(&range.hi)),
                ..range.clone()
            }),
            Pat::Reference(r) => Pat::Reference(syn::PatReference {
                pat: Box::new(self.transform_pattern(&r.pat)),
                ..r.clone()
            }),
            Pat::Slice(slice) => Pat::Slice(syn::PatSlice {
                elems: slice
                    .elems
                    .iter()
                    .map(|pat| self.transform_pattern(pat))
                    .collect(),
                ..slice.clone()
            }),
            Pat::Struct(s) => Pat::Struct(syn::PatStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldPat {
                        pat: Box::new(self.transform_pattern(&f.pat)),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Pat::Tuple(tuple) => Pat::Tuple(syn::PatTuple {
                elems: tuple
                    .elems
                    .iter()
                    .map(|pat| self.transform_pattern(pat))
                    .collect(),
                ..tuple.clone()
            }),
            Pat::TupleStruct(tuple) => Pat::TupleStruct(syn::PatTupleStruct {
                pat: syn::PatTuple {
                    elems: tuple
                        .pat
                        .elems
                        .iter()
                        .map(|pat| self.transform_pattern(pat))
                        .collect(),
                    ..tuple.pat.clone()
                },
                ..tuple.clone()
            }),
            Pat::Type(ty) => Pat::Type(syn::PatType {
                pat: Box::new(self.transform_pattern(&ty.pat)),
                ..ty.clone()
            }),
            _ => pat.clone(),
        }
    }
    fn transform_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Array(arr) => Expr::Array(syn::ExprArray {
                elems: arr
                    .elems
                    .iter()
                    .map(|expr| self.transform_expr(expr))
                    .collect(),
                ..arr.clone()
            }),
            Expr::Assign(assign) => Expr::Assign(syn::ExprAssign {
                left: Box::new(self.transform_expr(&*assign.left)),
                right: Box::new(self.transform_expr(&*assign.right)),
                ..assign.clone()
            }),
            Expr::AssignOp(assign) => Expr::AssignOp(syn::ExprAssignOp {
                left: Box::new(self.transform_expr(&*assign.left)),
                right: Box::new(self.transform_expr(&*assign.right)),
                ..assign.clone()
            }),
            Expr::Binary(binary) => Expr::Binary(syn::ExprBinary {
                left: Box::new(self.transform_expr(&*binary.left)),
                right: Box::new(self.transform_expr(&*binary.right)),
                ..binary.clone()
            }),
            Expr::Block(block) => Expr::Block(syn::ExprBlock {
                block: self.transform_block(&block.block),
                ..block.clone()
            }),
            Expr::Call(call) => Expr::Call(syn::ExprCall {
                func: Box::new(self.transform_expr(&*call.func)),
                args: call
                    .args
                    .iter()
                    .map(|expr| self.transform_expr(expr))
                    .collect(),
                ..call.clone()
            }),
            Expr::Cast(cast) => Expr::Cast(syn::ExprCast {
                expr: Box::new(self.transform_expr(&*cast.expr)),
                ..cast.clone()
            }),
            Expr::Field(field) => Expr::Field(syn::ExprField {
                base: Box::new(self.transform_expr(&*field.base)),
                ..field.clone()
            }),
            Expr::ForLoop(l) => Expr::ForLoop(syn::ExprForLoop {
                pat: self.transform_pattern(&l.pat),
                expr: Box::new(self.transform_expr(&*l.expr)),
                body: self.transform_block(&l.body),
                ..l.clone()
            }),
            Expr::Group(group) => Expr::Group(syn::ExprGroup {
                expr: Box::new(self.transform_expr(&group.expr)),
                ..group.clone()
            }),
            Expr::If(if_stmt) => Expr::If(syn::ExprIf {
                cond: Box::new(self.transform_expr(&if_stmt.cond)),
                then_branch: self.transform_block(&if_stmt.then_branch),
                else_branch: if_stmt.else_branch.as_ref().map(|(else_token, expr)| {
                    (else_token.clone(), Box::new(self.transform_expr(&expr)))
                }),
                ..if_stmt.clone()
            }),
            Expr::Index(index_expr) => Expr::Index(syn::ExprIndex {
                expr: Box::new(self.transform_expr(&*index_expr.expr)),
                index: Box::new(self.transform_expr(&*index_expr.index)),
                ..index_expr.clone()
            }),
            Expr::Let(l) => Expr::Let(syn::ExprLet {
                pat: self.transform_pattern(&l.pat),
                expr: Box::new(self.transform_expr(&*l.expr)),
                ..l.clone()
            }),
            Expr::Loop(loop_stmt) => Expr::Loop(syn::ExprLoop {
                body: self.transform_block(&loop_stmt.body),
                ..loop_stmt.clone()
            }),
            Expr::Macro(m) => self.macro_transform(m),
            Expr::Match(match_expr) => Expr::Match(syn::ExprMatch {
                expr: Box::new(self.transform_expr(&*match_expr.expr)),
                arms: match_expr
                    .arms
                    .iter()
                    .map(|arm| syn::Arm {
                        pat: self.transform_pattern(&arm.pat),
                        guard: arm.guard.as_ref().map(|(guard_token, expr)| {
                            (guard_token.clone(), Box::new(self.transform_expr(&expr)))
                        }),
                        body: Box::new(self.transform_expr(&arm.body)),
                        ..arm.clone()
                    })
                    .collect(),
                ..match_expr.clone()
            }),
            Expr::MethodCall(call) => Expr::MethodCall(syn::ExprMethodCall {
                receiver: Box::new(self.transform_expr(&call.receiver)),
                args: call
                    .args
                    .iter()
                    .map(|expr| self.transform_expr(expr))
                    .collect(),
                ..call.clone()
            }),
            Expr::Paren(paren) => Expr::Paren(syn::ExprParen {
                expr: Box::new(self.transform_expr(&paren.expr)),
                ..paren.clone()
            }),
            Expr::Range(range) => Expr::Range(syn::ExprRange {
                from: range
                    .from
                    .as_ref()
                    .map(|f| Box::new(self.transform_expr(&f))),
                to: range.to.as_ref().map(|t| Box::new(self.transform_expr(&t))),
                ..range.clone()
            }),
            Expr::Reference(reference) => Expr::Reference(syn::ExprReference {
                expr: Box::new(self.transform_expr(&reference.expr)),
                ..reference.clone()
            }),
            Expr::Repeat(repeat) => Expr::Repeat(syn::ExprRepeat {
                expr: Box::new(self.transform_expr(&repeat.expr)),
                len: Box::new(self.transform_expr(&repeat.len)),
                ..repeat.clone()
            }),
            Expr::Return(ret) => {
                if let Some(expr) = self.return_transform(ret) {
                    expr
                } else {
                    Expr::Return(syn::ExprReturn {
                        expr: ret.expr.as_ref().map(|e| Box::new(self.transform_expr(&e))),
                        ..ret.clone()
                    })
                }
            }
            Expr::Struct(s) => Expr::Struct(syn::ExprStruct {
                fields: s
                    .fields
                    .iter()
                    .map(|f| syn::FieldValue {
                        member: f.member.clone(),
                        expr: self.transform_expr(&f.expr),
                        ..f.clone()
                    })
                    .collect(),
                ..s.clone()
            }),
            Expr::Try(s) => Expr::Try(syn::ExprTry {
                expr: Box::new(self.transform_expr(&s.expr)),
                ..s.clone()
            }),
            Expr::Tuple(tuple) => Expr::Tuple(syn::ExprTuple {
                elems: tuple
                    .elems
                    .iter()
                    .map(|expr| self.transform_expr(expr))
                    .collect(),
                ..tuple.clone()
            }),
            Expr::Type(type_expr) => Expr::Type(syn::ExprType {
                expr: Box::new(self.transform_expr(&type_expr.expr)),
                ..type_expr.clone()
            }),
            Expr::Unary(unary) => Expr::Unary(syn::ExprUnary {
                expr: Box::new(self.transform_expr(&unary.expr)),
                ..unary.clone()
            }),
            Expr::Unsafe(unsafe_stmt) => Expr::Unsafe(syn::ExprUnsafe {
                block: self.transform_block(&unsafe_stmt.block),
                ..unsafe_stmt.clone()
            }),
            Expr::While(while_stmt) => Expr::While(syn::ExprWhile {
                cond: Box::new(self.transform_expr(&while_stmt.cond)),
                body: self.transform_block(&while_stmt.body),
                ..while_stmt.clone()
            }),
            Expr::Await(await_expr) => self.async_transform(await_expr),
            _ => expr.clone(),
        }
    }
    fn transform_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Local(local) => Stmt::Local(syn::Local {
                pat: self.transform_pattern(&local.pat),
                init: local.init.as_ref().map(|(eq_token, expr)| {
                    (eq_token.clone(), Box::new(self.transform_expr(&expr)))
                }),
                ..local.clone()
            }),
            Stmt::Expr(expr) => Stmt::Expr(self.transform_expr(&expr)),
            Stmt::Semi(expr, semi) => Stmt::Semi(self.transform_expr(expr), semi.clone()),
            _ => stmt.clone(),
        }
    }
    fn import(
        &mut self,
        input_tokens: &proc_macro2::TokenStream,
        is_image: bool,
    ) -> proc_macro2::TokenStream;
    fn async_transform(&mut self, input: &syn::ExprAwait) -> syn::Expr;
    fn macro_transform(&mut self, mac: &syn::ExprMacro) -> syn::Expr;
    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr>;
}
